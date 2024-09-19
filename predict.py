import os
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from cog import BasePredictor, Input, Path

CLIP_PATH = "google/siglip-so400m-patch14-384"
VLM_PROMPT = "A descriptive caption for this image:\n"
MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
    
    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class Predictor(BasePredictor):
    def setup(self):
        print("Loading CLIP")
        self.clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
        self.clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.clip_model.to("cuda")

        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, load_in_4bit=True, use_fast=False)

        print("Loading LLM")
        self.text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_4bit=True, device_map="auto", torch_dtype=torch.float16)
        self.text_model.eval()

        print("Loading image adapter")
        self.image_adapter = ImageAdapter(self.clip_model.config.hidden_size, self.text_model.config.hidden_size)
        self.image_adapter.load_state_dict(torch.load("image_adapter.pt", map_location="cpu"))
        self.image_adapter.eval()
        self.image_adapter.to("cuda")

    @torch.inference_mode()
    def predict(self, image: Path = Input(description="Input image to caption")) -> str:
        torch.cuda.empty_cache()

        # Preprocess image
        input_image = Image.open(image).convert("RGB")
        image = self.clip_processor(images=input_image, return_tensors='pt').pixel_values
        image = image.to('cuda')

        # Tokenize the prompt
        prompt = self.tokenizer.encode(VLM_PROMPT, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)

        # Embed image
        with autocast('cuda', enabled=True):
            vision_outputs = self.clip_model(pixel_values=image, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = self.image_adapter(image_features)
            embedded_images = embedded_images.to('cuda')
        
        # Embed prompt
        prompt_embeds = self.text_model.model.embed_tokens(prompt.to('cuda'))
        embedded_bos = self.text_model.model.embed_tokens(torch.tensor([[self.tokenizer.bos_token_id]], device=self.text_model.device, dtype=torch.int64))

        # Construct prompts
        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            prompt,
        ], dim=1).to('cuda')
        attention_mask = torch.ones_like(input_ids)

        generate_ids = self.text_model.generate(
            input_ids, 
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            max_new_tokens=300, 
            do_sample=True, 
            top_k=10, 
            temperature=0.5, 
            suppress_tokens=None
        )

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == self.tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]

        caption = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        return caption.strip()
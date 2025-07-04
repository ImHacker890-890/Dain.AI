import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel

class DainTextToImage(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        
        # CLIP для кодирования текста
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
        # U-Net архитектура (условная диффузия)
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=512,
        ).to(device)
        
        self.device = device

    def encode_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_embeddings = self.text_encoder(**inputs).last_hidden_state
        return text_embeddings

    def forward(self, noisy_images, timesteps, text_embeddings):
        return self.unet(noisy_images, timesteps, encoder_hidden_states=text_embeddings).sample

import torch
from PIL import Image
from .model import DainTextToImage
from diffusers import DDIMScheduler

def generate_from_prompt(prompt, num_images=1, steps=50, guidance_scale=7.5, device="cuda"):
    model = DainTextToImage(device).eval()
    model.load_state_dict(torch.load("dain_coco_epoch9.pt"))
    
    scheduler = DDIMScheduler(
        beta_start=0.0001,
        beta_end=0.02,
        num_train_timesteps=1000,
    )
    
    text_emb = model.encode_text([prompt] * num_images)
    latents = torch.randn((num_images, 4, 64, 64)).to(device)
    
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            noise_pred = model(latents, t, text_emb)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    images = (latents.clamp(-1, 1) + 1) / 2 * 255
    return [Image.fromarray(img.cpu().numpy().astype("uint8")) for img in images]

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import List, Union

def pil_to_tensor(image: Image.Image, device: str = "cuda") -> torch.Tensor:
    """Convert PIL Image to normalized torch tensor"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert torch tensor to PIL Image"""
    image_tensor = image_tensor.detach().cpu()
    image_tensor = (image_tensor + 1) / 2  # [-1, 1] -> [0, 1]
    image_tensor = image_tensor.clamp(0, 1)
    image = transforms.ToPILImage()(image_tensor.squeeze())
    return image

def show_images(images: Union[List[Image.Image], titles: List[str] = None, figsize=(10, 5)):
    """Display list of PIL Images"""
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img)
        plt.axis('off')
        if titles and i < len(titles):
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()

def image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    """Create a grid of PIL Images"""
    assert len(images) == rows * cols
    
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def add_noise(image: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, num_train_timesteps: int = 1000) -> torch.Tensor:
    """Add noise to images according to timestep (forward diffusion process)"""
    sqrt_alpha_prod = (1 - timesteps/num_train_timesteps).sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_prod = (timesteps/num_train_timesteps).sqrt().view(-1, 1, 1, 1)
    
    noisy_image = sqrt_alpha_prod * image + sqrt_one_minus_alpha_prod * noise
    return noisy_image

def save_model(model, path: str):
    """Save model weights"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path: str, device: str = "cuda"):
    """Load model weights"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model

def generate_latents(batch_size: int, height: int, width: int, device: str = "cuda") -> torch.Tensor:
    """Generate random latents for diffusion process"""
    return torch.randn(
        (batch_size, 4, height // 8, width // 8),  # Latent space is 8x smaller
        device=device
    )

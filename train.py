import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from .model import DainTextToImage
from tqdm import tqdm
import os

def load_coco_dataset(batch_size=8, image_size=64):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    dataset = CocoCaptions(
        root="data/train2017",
        annFile="data/annotations/captions_train2017.json",
        transform=transform,
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_on_coco(epochs=10, lr=1e-4, device="cuda"):
    model = DainTextToImage(device).train()
    dataloader = load_coco_dataset()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for images, captions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            text_emb = model.encode_text(captions)
            
            # Шумим изображения (прямой процесс диффузии)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, 1000, (images.shape[0],)).to(device)
            noisy_images = add_noise(images, noise, timesteps)
            
            # Предсказываем шум (обратный процесс)
            pred_noise = model(noisy_images, timesteps, text_emb)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"dain_coco_epoch{epoch}.pt")

# Dain - Diffussion image generator.
Dain - this is free open source model for image generation.
## Installation
Clone repository.
```bash
git clone https://github.com/ImHacker890-890/Dain.AI
cd Dain.AI
```
Install requirements.txt:
```bash
pip install -r requirements.txt
```
## Usage
Train the model.
```bash
python train.py
```
Generate image.
```py
from Dain import generate_from_prompt
images = generate_from_prompt(
    prompt="A cat sitting on a couch, oil painting style",
    num_images=4,
    steps=50,
    device="cuda"
)

for img in images:
    img.show()
```

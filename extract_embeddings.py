import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

print("Loading ResNet50...")

# Load pretrained ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Remove final classification layer
model = nn.Sequential(*list(model.children())[:-1])

model.eval()

print("ResNet50 Loaded Successfully!")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Image folder
images_folder = "data/fashion_images/images"

# Get all jpg images
image_files = sorted([
    f for f in os.listdir(images_folder)
    if f.lower().endswith(".jpg")
])

print(f"Total Images: {len(image_files)}")

embeddings = []

# Loop through all images
for image_name in tqdm(image_files):

    image_path = os.path.join(images_folder, image_name)

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        embedding = model(image)

    embedding = embedding.squeeze().numpy()

    row = [image_name.replace(".jpg", "")]

    row.extend(embedding.tolist())

    embeddings.append(row)

# Create dataframe

columns = ["image_id"]

for i in range(2048):
    columns.append(f"emb_{i+1}")

df = pd.DataFrame(embeddings, columns=columns)

df.to_csv("image_embeddings.csv", index=False)

print("===================================")
print("Embeddings Saved Successfully!")
print("File Name : image_embeddings.csv")
print("===================================")
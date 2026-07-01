import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity


# Load ResNet50

model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

model = nn.Sequential(
    *list(model.children())[:-1]
)

model.eval()


# Image Transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# Load Embeddings

embeddings = pd.read_csv(
    "image_embeddings.csv"
)


def recommend_similar_images(uploaded_file, top_k=5):

    image = Image.open(uploaded_file).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        embedding = model(image)

    embedding = embedding.squeeze().numpy()

    database = embeddings.iloc[:,1:].values

    similarity = cosine_similarity(
        [embedding],
        database
    )[0]

    indices = np.argsort(similarity)[::-1][:top_k]

    image_ids = embeddings.iloc[
        indices
    ]["image_id"].tolist()

    scores = similarity[indices]

    return image_ids, scores
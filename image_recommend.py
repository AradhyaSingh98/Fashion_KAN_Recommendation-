import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

print("Loading Image Embeddings...")

# Load embeddings
df = pd.read_csv("image_embeddings.csv")

# Image IDs
image_ids = df["image_id"].astype(str)

# Embedding vectors
embeddings = df.drop(columns=["image_id"])

print("Embeddings Loaded!")

# -------------------------------
# Query Image Index
# -------------------------------

query_index = 0

query_embedding = embeddings.iloc[query_index].values.reshape(1, -1)

# Cosine similarity
similarities = cosine_similarity(query_embedding, embeddings)[0]

# Top 6 (including query image itself)
top_indices = similarities.argsort()[-6:][::-1]

# Image folder
image_folder = "data/fashion_images/images"

# Plot
plt.figure(figsize=(18,5))

for i, idx in enumerate(top_indices):

    image_name = image_ids.iloc[idx] + ".jpg"

    image_path = os.path.join(image_folder, image_name)

    img = Image.open(image_path)

    plt.subplot(1,6,i+1)
    plt.imshow(img)
    plt.axis("off")

    if i == 0:
        plt.title("Query Image")
    else:
        plt.title(f"{similarities[idx]:.2f}")

plt.tight_layout()
plt.show()
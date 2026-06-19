import torch
import pandas as pd
from models.kan_model import create_model


# Load trained model

model = create_model()

model.load_state_dict(
    torch.load("fashion_kan_model.pth")
)

model.eval()


# Load dataset

data = pd.read_csv(
    "recommender_dataset.csv"
)


# Example User

user_age = int(
    input("Enter Age: ")
)
max_price = float(
    input("Enter Maximum Price: ")
)


# Candidate products

products = data[
[
    "article_id",
    "price",
    "product_type_name",
    "colour_group_name",
    "department_name"
]
].drop_duplicates()


# Take first 100 products for testing
products = products[
    products["price"] <= max_price
]

products = products.head(100)


# Create feature vectors

samples = []

for _, row in products.iterrows():

    samples.append(
        [
            user_age,
            row["price"],
            row["product_type_name"],
            row["colour_group_name"],
            row["department_name"]
        ]
    )


X = torch.tensor(
    samples,
    dtype=torch.float32
)


# Predict

with torch.no_grad():

    scores = model(X).squeeze()


# Attach scores

products["score"] = scores.numpy()
articles = pd.read_csv(
    "data/articles_hm.csv"
)

articles = articles[
[
    "article_id",
    "prod_name",
    "product_type_name",
    "colour_group_name",
    "department_name"
]
]

recommendations = products.merge(
    articles,
    on="article_id",
    how="left"
)


# Sort

recommendations = recommendations.sort_values(
    by="score",
    ascending=False
)


print("\nTop 10 Recommended Products:\n")

print(
    recommendations[
        [
            "prod_name",
            "product_type_name_y",
            "colour_group_name_y",
            "department_name_y",
            "score"
        ]
    ].head(10)
)
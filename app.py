import streamlit as st
import torch
import pandas as pd

from models.kan_model import create_model


st.title("Fashion Recommendation System")

st.write("KAN Based Fashion Recommender")


# Load model

model = create_model()

model.load_state_dict(
    torch.load("fashion_kan_model.pth")
)

model.eval()


# User Inputs

user_age = st.number_input(
    "Enter Age",
    min_value=10,
    max_value=100,
    value=25
)

max_price = st.number_input(
    "Maximum Price",
    min_value=0.0,
    value=0.05
)


if st.button("Recommend Products"):

    data = pd.read_csv(
        "recommender_dataset.csv"
    )

    products = data[
        [
            "article_id",
            "price",
            "product_type_name",
            "colour_group_name",
            "department_name"
        ]
    ].drop_duplicates()

    products = products[
        products["price"] <= max_price
    ]

    products = products.head(100)

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

    with torch.no_grad():

        scores = model(X).squeeze()

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

    recommendations = recommendations.sort_values(
        by="score",
        ascending=False
    )

    st.subheader("Top 10 Recommendations")

    st.dataframe(
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
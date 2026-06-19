import pandas as pd

articles = pd.read_csv(
    "data/articles_hm.csv"
)

print(articles.columns.tolist())
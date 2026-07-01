import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load data

customers = pd.read_csv(
"data/customer_hm.csv"
)

articles = pd.read_csv(
"data/articles_hm.csv"
)

transactions = pd.read_csv(
"data/transactions_hm.csv"
)

# Merge

data = transactions.merge(
customers,
on="customer_id",
how="left"
)

data = data.merge(
articles,
on="article_id",
how="left"
)

# Select features

data = data[
[
"customer_id",
"article_id",
"age",
"price",
"product_type_name",
"colour_group_name",
"department_name"
]
]

# Fill missing age

data["age"] = data["age"].fillna(
data["age"].mean()
)

# Label Encoding

encoder = LabelEncoder()

data["product_type_name"] = encoder.fit_transform(
data["product_type_name"]
)

data["colour_group_name"] = encoder.fit_transform(
data["colour_group_name"]
)

data["department_name"] = encoder.fit_transform(
data["department_name"]
)

# Feature Scaling

scaler = MinMaxScaler()

feature_cols = [
"age",
"price",
"product_type_name",
"colour_group_name",
"department_name"
]

data[feature_cols] = scaler.fit_transform(
data[feature_cols]
)

print(data.head())

print(data.dtypes)

# Save processed data

data.to_csv(
"processed_data.csv",
index=False
)

print("Data Saved Successfully")

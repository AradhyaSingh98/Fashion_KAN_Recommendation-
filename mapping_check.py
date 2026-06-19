import pandas as pd

articles = pd.read_csv("data/articles_hm.csv")

print("\nProduct Types:")
print(articles["product_type_name"].unique()[:20])

print("\nColours:")
print(articles["colour_group_name"].unique()[:20])

print("\nDepartments:")
print(articles["department_name"].unique()[:20])
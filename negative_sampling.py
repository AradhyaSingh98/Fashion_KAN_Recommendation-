import pandas as pd
import numpy as np


# Load data

data = pd.read_csv("processed_data.csv")


# Positive samples

positive = data.copy()

positive["target"] = 1


# Negative samples

negative = data.copy()

negative["article_id"] = np.random.permutation(
    negative["article_id"].values
)

negative["target"] = 0


# Combine

final_data = pd.concat(
    [positive, negative],
    ignore_index=True
)


# Shuffle

final_data = final_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Save

final_data.to_csv(
    "recommender_dataset.csv",
    index=False
)

print(final_data.head())

print("\nShape:")
print(final_data.shape)
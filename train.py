import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from models.kan_model import create_model
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Load recommender dataset

data = pd.read_csv(
    "recommender_dataset.csv"
)


# Hardware ke hisab se sample

data = data.sample(
    10000,
    random_state=42
)


# Features

X = data[
[
    "age",
    "price",
    "product_type_name",
    "colour_group_name",
    "department_name"
]
]


# Target

y = data["target"]


# Tensor conversion

X = torch.tensor(
    X.values,
    dtype=torch.float32
)

y = torch.tensor(
    y.values,
    dtype=torch.float32
)


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Train Shape:")
print(X_train.shape)

print("Test Shape:")
print(X_test.shape)


# Model

model = create_model()

print(model)


# Loss

criterion = nn.BCEWithLogitsLoss()


# Optimizer

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training

epochs = 5

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(X_train)

    loss = criterion(
        output.squeeze(),
        y_train
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}"
    )


# Testing

with torch.no_grad():

    outputs = model(X_test)

    probs = torch.sigmoid(
        outputs.squeeze()
    )

    preds = (
        probs > 0.5
    ).float()

    y_true = y_test.numpy()
    y_pred = preds.numpy()

    print(
        f"Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%"
    )

    print(
        f"Precision: {precision_score(y_true, y_pred):.4f}"
    )

    print(
        f"Recall: {recall_score(y_true, y_pred):.4f}"
    )

    print(
        f"F1 Score: {f1_score(y_true, y_pred):.4f}"
    )
import pandas as pd
import torch
from models.kan_model import create_model


# Load processed data

data = pd.read_csv(
    "processed_data.csv"
)
data = data.sample(10000, random_state=42)


# X = Training Data

X = data[
[
"age",
"price",
"product_type_name",
"colour_group_name",
"department_name"
]
]


# y = Target Data

y = [1] * len(data)



# Convert X into tensor

X = torch.tensor(
    X.values,
    dtype=torch.float32
)


# Convert y into tensor

y = torch.tensor(
    y,
    dtype=torch.float32
)



print("Training Data Shape:")
print(X.shape)


print("Target Shape:")
print(y.shape)
# Create KAN model

model = create_model()


print(model)


print("KAN model created")
import torch.nn as nn


# Loss function

criterion = nn.MSELoss()


# Optimizer

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training

epochs = 5


for epoch in range(epochs):

    optimizer.zero_grad()


    output = model(X)


    loss = criterion(
        output.squeeze(),
        y
    )


    loss.backward()


    optimizer.step()


    print(
        f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}"
    )
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def proc_col(col, train_col=None):
    """Encodes a pandas column with continuous ids."""
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o: i for i, o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    """Encodes rating data with continuous user and movie ids.
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _, col, _ = proc_col(df[col_name], train_col)
        df[col_name] = col
    return df

class MF_with_known_feature(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, known_feature_size=1):
        super(MF_with_known_feature, self).__init__()

        # Embeddings with learnable parameters
        self.user_embedding_learnable = nn.Embedding(num_users + 1, emb_size)
        self.item_embedding_learnable = nn.Embedding(num_items + 1, emb_size)
        self.user_embedding_known_feature = nn.Embedding(num_users + 1, known_feature_size)

        # Initialize learnable embeddings with uniform random values
        self.user_embedding_learnable.weight.data.uniform_(0, 0.05)
        self.item_embedding_learnable.weight.data.uniform_(0, 0.05)

    def forward(self, u, v, known_feature):
        u_learnable = self.user_embedding_learnable(u)
        v_learnable = self.item_embedding_learnable(v)

        # Retrieve the known feature embedding for the user
        known_feature_embedding = self.user_embedding_known_feature(u)

        # Combine the known feature with the learnable user embedding
        u_combined = u_learnable + known_feature_embedding

        return (u_combined * v_learnable).sum(1)

def train_epocs(model, epochs=10, lr=0.1, wd=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()

    for i in range(epochs):
        users = torch.LongTensor(df_train.userId.values)
        items = torch.LongTensor(df_train.movieId.values)
        known_feature = torch.FloatTensor(df_train.age.values)  # Assuming age is the known feature
        ratings = torch.FloatTensor(df_train.rating.values)

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        y_hat = model(users, items, known_feature)
        loss = F.mse_loss(y_hat, ratings)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        print(f'Epoch {i + 1}, Loss: {loss.item()}')

# Create a sample dataset
data = pd.DataFrame({
    'userId': [1, 2, 3, 1, 2, 3],
    'movieId': [101, 102, 103, 101, 102, 103],
    'rating': [5.0, 4.0, 3.5, 4.5, 3.0, 2.5],
    'age': [25, 30, 22, 25, 30, 22]  # Known feature (e.g., user age)
})

# Split the data into training and validation sets
msk = np.random.rand(len(data)) < 0.8
train = data[msk].copy()
val = data[~msk].copy()

# Encode the datasets
df_train = encode_data(train)
df_val = encode_data(val, train)

num_users = data['userId'].nunique()
num_items = data['movieId'].nunique()

# Create an instance of the model
model = MF_with_known_feature(num_users, num_items, emb_size=100, known_feature_size=1)

# Train the model
train_epocs(model, epochs=30, lr=0.1)
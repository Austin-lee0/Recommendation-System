# this is my recommendation system pt 1!!!!

from pathlib import Path
import pandas as pd
import numpy as np
import torch # apparently pytorch doesn't download with most recent python verison - need to downgrade interpreter
import torch.nn as nn
import torch.nn.functional as F

data = pd.read_csv("/Users/austinblee/Downloads/ml-latest-small/ratings.csv")
data.head()

print(data.head())

np.random.seed(3)
msk = np.random.rand(len(data)) < 0.8
train = data[msk].copy()
val = data[~msk].copy()


# here is a handy function modified from fast.ai
# proc_col encodes pandas column with continuous ids ->  [10,20,30] ->
# {}
def proc_col(col, train_col=None):  # train_col for training dataset
    """Encodes a pandas column with continuous ids.
   """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    # iterates over the pairs i,o and returns elements from o and corresponding indices from i
    name2idx = {o: i for i, o in enumerate(uniq)}
    # ex: dictionary for uniq [10,20,30] would return {10:0,20:1,30:2}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)
# returns dictionary of elements and indices
# returns encoded index values on the initial column
# returns number of unique ids


# returns dictionary of elements and indices
# returns encoded index values on the initial column
# returns number of unique ids

def encode_data(df, train=None):
    """Encodes rating data with continuous user and movie ids.
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
# converts unique userids or movieids into continuous
# integers using proc_col function and stores in _,col,_ (gets the second return value of the proc_col function)
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df


# encodes the data sets
print(train.head())
df_train = encode_data(train)
df_val = encode_data(val, train)
print(df_train.head())


class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=85):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users+1, emb_size)
        self.item_emb = nn.Embedding(num_items+1, emb_size)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)

    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)
        return (u * v).sum(1)


#for demonstration purposes:

'''
# Example: Consider 3 users, 4 items, and an embedding size of 2
num_users = 3
num_items = 4
emb_size = 2

# Create an instance of the matrix factorization model
model = nn.Embedding(num_users, emb_size), nn.Embedding(num_items, emb_size)

# Assume training data: user 1 interacts with item 2 with a rating of 4.0
user_index = torch.tensor([1])
item_index = torch.tensor([2])
rating = torch.tensor([4.0])

# Forward pass: calculate the predicted rating
user_embedding = model[0](user_index)  # Embedding for user 1
item_embedding = model[1](item_index)  # Embedding for item 2
predicted_rating = (user_embedding * item_embedding).sum(1)

# Print the results
print("User Embedding Matrix:")
print(user_embedding)

print("\nItem Embedding Matrix:")
print(item_embedding)

print("\nElement-wise Multiplication Matrix:")
print(user_embedding * item_embedding)

print("\nPredicted Rating:")
print(predicted_rating)
'''

num_users = len(df_train.userId.unique())
num_items = len(df_train.movieId.unique())
model = MF(num_users, num_items, emb_size=100)


def train_epocs(model, epochs=10, lr=0.1, wd=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for i in range(epochs):
        users = torch.LongTensor(df_train.userId.values)  # .cuda()
        items = torch.LongTensor(df_train.movieId.values)  # .cuda()
        ratings = torch.FloatTensor(df_train.rating.values)  # .cuda()
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
    test_loss(model)


def test_loss(model):
    model.eval()
    users = torch.LongTensor(df_val.userId.values)  # .cuda()
    items = torch.LongTensor(df_val.movieId.values)  # .cuda()
    ratings = torch.FloatTensor(df_val.rating.values)  # .cuda()
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())


train_epocs(model, epochs=30, lr=0.1)

    # Print the maximum indices in the validation set





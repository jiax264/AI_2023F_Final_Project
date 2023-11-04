import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# # read data and apply one-hot encoding
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# data = pd.read_csv(url, header=None)
# X = data.iloc[:, 0:4]
# y = data.iloc[:, 4:]
# ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
# y = ohe.transform(y)
#
# # convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
# X = torch.tensor(X.values, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32)
#
# # split
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)



train_df = pd.read_csv('../03_Data_for_Modeling/train.csv')

X_train = train_df.drop(columns=['time_window'])
y_train = train_df['time_window']

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y_train.values.reshape(-1, 1))
y_train = ohe.transform(y_train.values.reshape(-1, 1))

# check one hot-encoding
unique_categories = ohe.categories_[0]
unique_categories_list = unique_categories.tolist()

for index, category in enumerate(unique_categories_list):
    one_hot_encoded_value = [0] * len(unique_categories_list)
    one_hot_encoded_value[index] = 1
    print(f'{category}: {one_hot_encoded_value}')

valid_df = pd.read_csv('../03_Data_for_Modeling/valid.csv')

X_test = valid_df.drop(columns=['time_window'])
y_test = valid_df['time_window']

y_test = ohe.transform(y_test.values.reshape(-1, 1))

X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(24, 256)
        self.hidden2 = nn.Linear(256, 1024)
        self.hidden3 = nn.Linear(1024, 256)
        self.hidden4 = nn.Linear(256, 64)
        self.act = nn.ReLU()
        self.output = nn.Linear(64, 4)

    def forward(self, x):
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.act(self.hidden3(x))
        x = self.act(self.hidden4(x))
        x = self.output(x)
        return x

# loss metric and optimizer
model = Multiclass()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# prepare model and training parameters
n_epochs = 300
batch_size = 64
batches_per_epoch = len(X_train) // batch_size

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

# training loop
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    # set model in training mode and run through each batch
    model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
    # set model in evaluation mode and run through the test set
    model.eval()
    y_pred = model(X_test)
    ce = loss_fn(y_pred, y_test)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
    ce = float(ce)
    acc = float(acc)
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(ce)
    test_acc_hist.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")

# Restore best model
model.load_state_dict(best_weights)

# Plot the loss and accuracy
plt.plot(train_loss_hist, label="train")
plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.show()

plt.plot(train_acc_hist, label="train")
plt.plot(test_acc_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()
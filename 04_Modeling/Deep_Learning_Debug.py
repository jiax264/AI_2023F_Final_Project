import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, TensorDataset

# data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv", header=None)
# X = data.iloc[:, 0:4]
# y = data.iloc[:, 4:]
# label_encoder = LabelEncoder()
#
# # Fit and transform the labels into integers
# y_encoded = label_encoder.fit_transform(y)
#
# # Convert pandas DataFrame (X) and numpy array (y_encoded) into PyTorch tensors
# X_tensor = torch.tensor(X.values, dtype=torch.float32)
# y_tensor = torch.tensor(y_encoded, dtype=torch.long)
#
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, train_size=0.7, shuffle=True)
# class Multiclass(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(4, 8)
#         self.act = nn.ReLU()
#         self.output = nn.Linear(8, 3)
#
#     def forward(self, x):
#         x = self.act(self.hidden(x))
#         x = self.output(x)
#         return x

#
# train_df = pd.read_csv('../03_Data_for_Modeling/train_with_avg_hour_from_4_nearest_neigh.csv')
# train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)
# train_df_subset = train_df.iloc[:1000]
# X_train = train_df_subset.drop(columns=['time_window'])
# y_train = train_df_subset['time_window']
# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(y_train)
# X_train = torch.tensor(X_train.values, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)
#
# valid_df = pd.read_csv('../03_Data_for_Modeling/valid_with_avg_hour_from_4_nearest_neigh.csv')
# valid_df = valid_df.sample(frac=1, random_state=1).reset_index(drop=True)
# valid_df_subset = valid_df.iloc[:100]
# X_test = valid_df_subset.drop(columns=['time_window'])
# y_test = valid_df_subset['time_window']
# y_test = label_encoder.transform(y_test)
# X_test = torch.tensor(X_test.values, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.long)

#############

train_df = pd.read_csv('../03_Data_for_Modeling/train_with_avg_hour_from_4_nearest_neigh.csv')

X_train = train_df.drop(columns=['time_window'])
y_train = train_df['time_window']

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['time_window'])

valid_df = pd.read_csv('../03_Data_for_Modeling/valid_with_avg_hour_from_4_nearest_neigh.csv')

X_test = valid_df.drop(columns=['time_window'])
y_test = valid_df['time_window']

y_test = label_encoder.transform(y_test)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train.numpy())
sample_weights = torch.tensor([class_weights[t] for t in y_train])
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(25, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.act = nn.ReLU()
        self.output = nn.Linear(64, 4)

    def forward(self, x):
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.output(x)
        return x

# loss metric and optimizer
model = Multiclass()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# prepare model and training parameters
n_epochs = 100
batch_size = 500
batches_per_epoch = len(X_train) // batch_size

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

# Create TensorDatasets for your train and test sets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders with the datasets and the sampler for the training set
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    sampler=sampler  # Note that we're using sampler here
)
# For the test set, you don't need to use the sampler
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# training loop
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []

    # set model in training mode for the training phase
    model.train()
    train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}', leave=False)
    for X_batch, y_batch in train_bar:
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        # compute and store metrics
        acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
        epoch_loss.append(loss.item())
        epoch_acc.append(acc.item())
        train_bar.set_postfix(loss=np.mean(epoch_loss), acc=np.mean(epoch_acc))

    # set model in evaluation mode for the validation phase
    model.eval()
    val_loss = []
    val_acc = []
    val_bar = tqdm(test_loader, desc=f'Validation Epoch {epoch}', leave=False)
    with torch.no_grad():
        for X_batch, y_batch in val_bar:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
            val_loss.append(loss.item())
            val_acc.append(acc.item())
            val_bar.set_postfix(loss=np.mean(val_loss), acc=np.mean(val_acc))

    # Calculate average loss and accuracy over an epoch
    train_epoch_loss = np.mean(epoch_loss)
    train_epoch_acc = np.mean(epoch_acc)
    val_epoch_loss = np.mean(val_loss)
    val_epoch_acc = np.mean(val_acc)

    # Append to history for plotting later
    train_loss_hist.append(train_epoch_loss)
    train_acc_hist.append(train_epoch_acc)
    test_loss_hist.append(val_epoch_loss)
    test_acc_hist.append(val_epoch_acc)

    # Check if the current epoch's accuracy is the best one
    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        best_weights = copy.deepcopy(model.state_dict())

    # Display metrics at the end of each epoch
    print(f'Epoch {epoch} Training: Loss: {train_epoch_loss:.4f}, Accuracy: {train_epoch_acc * 100:.2f}%')
    print(f'Epoch {epoch} Validation: Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc * 100:.2f}%')

# Restore best model
model.load_state_dict(best_weights)

# Plot the loss and accuracy
plt.plot(train_loss_hist, label="train")
plt.plot(test_loss_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

plt.plot(train_acc_hist, label="train")
plt.plot(test_acc_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

# training loop
# for epoch in range(n_epochs):
#     epoch_loss = []
#     epoch_acc = []
#     # set model in training mode and run through each batch
#     model.train()
#     with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
#         bar.set_description(f"Epoch {epoch}")
#         for i in bar:
#             # take a batch
#             start = i * batch_size
#             X_batch = X_train[start:start+batch_size]
#             y_batch = y_train[start:start+batch_size]
#             # forward pass
#             y_pred = model(X_batch)
#             loss = loss_fn(y_pred, y_batch)
#             # backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             # update weights
#             optimizer.step()
#             # compute and store metrics
#             acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
#             epoch_loss.append(float(loss))
#             epoch_acc.append(float(acc))
#             bar.set_postfix(
#                 loss=float(loss),
#                 acc=float(acc)
#             )
#     # set model in evaluation mode and run through the test set
#     model.eval()
#     y_pred = model(X_test)
#     ce = loss_fn(y_pred, y_test)
#     acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
#     ce = float(ce)
#     acc = float(acc)
#     train_loss_hist.append(np.mean(epoch_loss))
#     train_acc_hist.append(np.mean(epoch_acc))
#     test_loss_hist.append(ce)
#     test_acc_hist.append(acc)
#     if acc > best_acc:
#         best_acc = acc
#         best_weights = copy.deepcopy(model.state_dict())
#     print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")
#
# # Restore best model
# model.load_state_dict(best_weights)
#
# # Plot the loss and accuracy
# plt.plot(train_loss_hist, label="train")
# plt.plot(test_loss_hist, label="test")
# plt.xlabel("epochs")
# plt.ylabel("cross entropy")
# plt.legend()
# plt.show()
#
# plt.plot(train_acc_hist, label="train")
# plt.plot(test_acc_hist, label="test")
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.legend()
# plt.show()


# Define the neural network model
# class Multiclass(nn.Module):
#     def __init__(self, input_size, layer_sizes):
#         super().__init__()
#         layers = []
#         for size in layer_sizes:
#             layers.append(nn.Linear(input_size, size))
#             layers.append(nn.ReLU())
#             input_size = size  # Next layer's input is current layer's output size
#         layers.append(nn.Linear(input_size, 4))  # Output layer
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.layers(x)
#
# # Read the data
# train_df = pd.read_csv('../03_Data_for_Modeling/train_with_avg_hour_from_4_nearest_neigh.csv')
# valid_df = pd.read_csv('../03_Data_for_Modeling/valid_with_avg_hour_from_4_nearest_neigh.csv')
#
# # Prepare the data
# X_train = train_df.drop(columns=['time_window']).values
# y_train = train_df['time_window'].values
# X_test = valid_df.drop(columns=['time_window']).values
# y_test = valid_df['time_window'].values
#
# # Encoding the labels
# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(y_train)
# y_test = label_encoder.transform(y_test)
#
# # Convert data to PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.long)
#
# # Define hyperparameters to tune
# hidden_layers_options = [[128], [128, 64], [128, 64, 32]]
# learning_rate_options = [0.1, 0.01, 0.001]
# batch_size_options = [16, 32, 64]
#
# # Define the training parameters
# n_epochs = 100
# loss_fn = nn.CrossEntropyLoss()
#
# # Record the best hyperparameters
# best_hyperparams = {
#     'hidden_layers': None,
#     'learning_rate': None,
#     'batch_size': None,
#     'validation_accuracy': float('-inf'),
#     'model_state': None
# }
#
# # Hyperparameter tuning
# for hidden_layers in hidden_layers_options:
#     for learning_rate in learning_rate_options:
#         for batch_size in batch_size_options:
#
#             # Initialize model, loss, and optimizer with current set of hyperparameters
#             model = Multiclass(X_train.shape[1], hidden_layers)
#             optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#             # Training loop
#             for epoch in range(n_epochs):
#                 permutation = torch.randperm(X_train.size()[0])
#                 for i in range(0, X_train.size()[0], batch_size):
#                     indices = permutation[i:i+batch_size]
#                     batch_x, batch_y = X_train[indices], y_train[indices]
#
#                     optimizer.zero_grad()
#                     outputs = model(batch_x)
#                     loss = loss_fn(outputs, batch_y)
#                     loss.backward()
#                     optimizer.step()
#
#             # Validation
#             with torch.no_grad():
#                 outputs = model(X_test)
#                 _, predicted = torch.max(outputs, 1)
#                 correct = (predicted == y_test).float().sum()
#                 validation_accuracy = correct / y_test.size(0)
#
#             # If the current model is the best, save the hyperparameters and model state
#             if validation_accuracy > best_hyperparams['validation_accuracy']:
#                 best_hyperparams['hidden_layers'] = hidden_layers
#                 best_hyperparams['learning_rate'] = learning_rate
#                 best_hyperparams['batch_size'] = batch_size
#                 best_hyperparams['validation_accuracy'] = validation_accuracy
#                 best_hyperparams['model_state'] = copy.deepcopy(model.state_dict())
#
# # Output the best hyperparameters
# print(f"Best Hyperparameters:\n"
#       f"Hidden Layers: {best_hyperparams['hidden_layers']}\n"
#       f"Learning Rate: {best_hyperparams['learning_rate']}\n"
#       f"Batch Size: {best_hyperparams['batch_size']}\n"
#       f"Validation Accuracy: {best_hyperparams['validation_accuracy']:.4f}")
#
# # Load the best model
# model.load_state_dict(best_hyperparams['model_state'])

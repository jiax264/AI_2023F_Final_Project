import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import precision_recall_fscore_support


train_df = pd.read_csv('../03_Data_for_Modeling/train_mlp_new_time_window.csv')
X_train = train_df.drop(columns=['time_window'])
y_train = train_df['time_window']

valid_df = pd.read_csv('../03_Data_for_Modeling/val_mlp_new_time_window.csv')
X_val = valid_df.drop(columns=['time_window'])
y_val = valid_df['time_window']

test_df = pd.read_csv('../03_Data_for_Modeling/test_mlp_new_time_window.csv')
X_test = test_df.drop(columns=['time_window'])
y_test = test_df['time_window']

X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train.numpy())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
sample_weights = torch.tensor([class_weights[t] for t in y_train])
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(25, 64)
        # self.dropout1 = nn.Dropout(0.1)
        self.hidden2 = nn.Linear(64, 64)
        # self.dropout2 = nn.Dropout(0.1)
        self.act = nn.ReLU()
        # self.act = nn.LeakyReLU(0.01)
        self.output = nn.Linear(64, 4)

    def forward(self, x):
        x = self.act(self.hidden1(x))
        # x = self.dropout1(self.act(self.hidden1(x)))
        x = self.act(self.hidden2(x))
        # x = self.dropout2(self.act(self.hidden2(x)))
        x = self.output(x)
        return x


# loss metric and optimizer
model = Multiclass()
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# prepare model and training parameters
n_epochs = 300
batch_size = 500
batches_per_epoch = len(X_train) // batch_size

best_acc = - np.inf  # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

# Create TensorDatasets for your train and test sets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders with the datasets and the sampler for the training set
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    sampler=sampler
)
# For the test set, you don't need to use the sampler
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# training loop
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    model.train()
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
        epoch_loss.append(loss.item())
        epoch_acc.append(acc.item())

    model.eval()
    val_loss = []
    val_acc = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
            val_loss.append(loss.item())
            val_acc.append(acc.item())

    # Calculate average loss and accuracy over an epoch
    train_epoch_loss = np.mean(epoch_loss)
    train_epoch_acc = np.mean(epoch_acc)
    val_epoch_loss = np.mean(val_loss)
    val_epoch_acc = np.mean(val_acc)

    # Append to history for plotting later
    train_loss_hist.append(train_epoch_loss)
    train_acc_hist.append(train_epoch_acc)
    val_loss_hist.append(val_epoch_loss)
    val_acc_hist.append(val_epoch_acc)

    # Check if the current epoch's accuracy is the best one
    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        best_weights = copy.deepcopy(model.state_dict())

# Restore best model
model.load_state_dict(best_weights)

# Plot the loss and accuracy
plt.figure()
plt.plot(train_loss_hist, label="train")
plt.plot(val_loss_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig('loss_mlp_2hidden_64_64_300epoch_500batch_001lr_weight_decay_1e-5.png')

plt.figure()
plt.plot(train_acc_hist, label="train")
plt.plot(val_acc_hist, label="validation")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig('accuracy_mlp_2hidden_64_64_300epoch_500batch_001lr_weight_decay_1e-5.png')

model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(y_batch.numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Validation')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_true)))
plt.xticks(tick_marks, np.unique(y_true))
plt.yticks(tick_marks, np.unique(y_true))

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('cm_val_mlp_2hidden_64_64_300epoch_500batch_001lr_weight_decay_1e-5.png')

y_test_pred = []
y_test_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_test_pred.extend(predicted.numpy())
        y_test_true.extend(y_batch.numpy())

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test_true, y_test_pred)
plt.figure(figsize=(10, 7))
plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Test')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_test_true)))
plt.xticks(tick_marks, np.unique(y_test_true))
plt.yticks(tick_marks, np.unique(y_test_true))

thresh = cm_test.max() / 2.
for i, j in itertools.product(range(cm_test.shape[0]), range(cm_test.shape[1])):
    plt.text(j, i, format(cm_test[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm_test[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('cm_test_mlp_2hidden_64_64_300epoch_500batch_001lr_weight_decay_1e-5.png')

class_names = ['rush_morning', 'rush_evening', 'non_rush_day', 'non_rush_night']

precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

print("Validation Set Class-wise Metrics:")
for i, class_name in enumerate(class_names):
    print(f"{class_name} - Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1-Score: {f1_score[i]:.2f}")

precision_macro, recall_macro, f1_score_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
precision_micro, recall_micro, f1_score_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("\nMacro Averages:")
print(f"Precision: {precision_macro:.2f}, Recall: {recall_macro:.2f}, F1-Score: {f1_score_macro:.2f}")

print("\nMicro Averages:")
print(f"Precision: {precision_micro:.2f}, Recall: {recall_micro:.2f}, F1-Score: {f1_score_micro:.2f}")

print("\nTest Set Class-wise Metrics:")
precision_test, recall_test, f1_score_test, _ = precision_recall_fscore_support(y_test_true, y_test_pred, average=None)

for i, class_name in enumerate(class_names):
    print(
        f"{class_name} - Precision: {precision_test[i]:.2f}, Recall: {recall_test[i]:.2f}, F1-Score: {f1_score_test[i]:.2f}")

precision_macro_test, recall_macro_test, f1_score_macro_test, _ = precision_recall_fscore_support(y_test_true,
                                                                                                  y_test_pred,
                                                                                                  average='macro')
precision_micro_test, recall_micro_test, f1_score_micro_test, _ = precision_recall_fscore_support(y_test_true,
                                                                                                  y_test_pred,
                                                                                                  average='micro')

print("\nTest Set Macro Averages:")
print(f"Precision: {precision_macro_test:.2f}, Recall: {recall_macro_test:.2f}, F1-Score: {f1_score_macro_test:.2f}")

print("\nTest Set Micro Averages:")
print(f"Precision: {precision_micro_test:.2f}, Recall: {recall_micro_test:.2f}, F1-Score: {f1_score_micro_test:.2f}")


# Uncomment below for probabilities for each class for ensemble
# def get_probabilities(model, data_loader):
#     model.eval()
#     probabilities = []
#
#     with torch.no_grad():
#         for X_batch, _ in data_loader:
#             outputs = model(X_batch)
#             probabilities_batch = torch.nn.functional.softmax(outputs, dim=1)
#             probabilities.extend(probabilities_batch.numpy())
#
#     return np.array(probabilities)
#
#
# # Obtain probabilities for validation and test sets
# val_probabilities_mlp = get_probabilities(model, val_loader)
# test_probabilities_mlp = get_probabilities(model, test_loader)
#
# # Convert to DataFrames
# val_probabilities_mlp_df = pd.DataFrame(val_probabilities_mlp, columns=class_names)
# test_probabilities_mlp_df = pd.DataFrame(test_probabilities_mlp, columns=class_names)
#
# val_probabilities_mlp_df.to_csv("../03_Data_for_Modeling/val_mlp_ensemble_probabilities.csv", index=False)
# test_probabilities_mlp_df.to_csv("../03_Data_for_Modeling/test_mlp_ensemble_probabilities.csv", index=False)

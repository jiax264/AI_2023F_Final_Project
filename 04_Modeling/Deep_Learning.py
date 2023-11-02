import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



train_df = pd.read_csv('../03_Data_for_Modeling/train.csv')

# Split features and target for training data
X_train = train_df.drop(columns=['time_window'])
y_train = train_df['time_window']

# Load validation data
valid_df = pd.read_csv('../03_Data_for_Modeling/valid.csv')

# Split features and target for validation data
X_val = valid_df.drop(columns=['time_window'])
y_val = valid_df['time_window']

# Convert target to integer labels
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)  # Use transform for validation data to ensure consistency

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val.values)
y_val_tensor = torch.LongTensor(y_val)


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # 4 classes: morning, afternoon, night, late night

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = FeedforwardNN(X_train.shape[1])

# Parameters
epochs = 10
batch_size = 64
learning_rate = 0.001

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training loop
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in tqdm(train_loader):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader)}')


def evaluate_model(model, loader, dataset_type="Training"):
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            _, predictions = torch.max(outputs, 1)
            true_labels.extend(y_batch)
            predicted_labels.extend(predictions)

    true_labels = [label.item() for label in true_labels]
    predicted_labels = [label.item() for label in predicted_labels]

    acc = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    cm = confusion_matrix(true_labels, predicted_labels)

    print(f"{dataset_type} Accuracy: {acc:.4f}")
    print(f"{dataset_type} Precision: {precision:.4f}")
    print(f"{dataset_type} Recall: {recall:.4f}")
    print(f"{dataset_type} F1 Score: {f1:.4f}")
    print()

    # Plotting confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{dataset_type} Confusion Matrix')
    plt.show()

    return acc, precision, recall, f1

print("Evaluating on Training Data...")
train_acc, train_precision, train_recall, train_f1 = evaluate_model(model, train_loader, dataset_type="Training")

print("Evaluating on Validation Data...")
val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, dataset_type="Validation")

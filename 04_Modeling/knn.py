#!/usr/bin/env python
# coding: utf-8


import random
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import BallTree
from scipy.stats import mode
from sklearn.model_selection import train_test_split

# In[16]:


has_time_df = pd.read_csv("../03_Data_for_Modeling/knn.csv")

# In[17]:


train_val, test = train_test_split(has_time_df, test_size=0.1, stratify=has_time_df['time_window'], random_state=42)
train, val = train_test_split(train_val, test_size=2 / 9, stratify=train_val['time_window'], random_state=42)

# In[18]:


tree = BallTree(train[['latitude', 'longitude']].values)

k_value = 4


def calculate_mode_time_window_of_nearest(train_tree, train_data, point, k=k_value):
    dist, ind = train_tree.query(point, k=k)
    time_windows = train_data.iloc[ind[0]]['time_window']

    if len(time_windows) < k:
        return np.nan

    mode_result = mode(time_windows)
    modes = mode_result.mode

    if modes.size == 0:  # No mode found
        return np.nan
    elif modes.size > 1:  # Handle ties by selecting a random mode
        return random.choice(modes)
    else:
        return modes.item()


has_time_df['most_common_time_window_{}_neigh'.format(k_value)] = has_time_df.apply(
    lambda row: calculate_mode_time_window_of_nearest(
        tree,
        train[['time_window']],
        np.array([[row['latitude'], row['longitude']]]),
    ),
    axis=1
)

df_with_avg_hour = has_time_df.set_index('master_record_number')[['most_common_time_window_{}_neigh'.format(k_value)]]

train_with_avg_hour = train.set_index('master_record_number').join(df_with_avg_hour)
val_with_avg_hour = val.set_index('master_record_number').join(df_with_avg_hour)
test_with_avg_hour = test.set_index('master_record_number').join(df_with_avg_hour)

train_with_avg_hour = train_with_avg_hour.reset_index()
val_with_avg_hour = val_with_avg_hour.reset_index()
test_with_avg_hour = test_with_avg_hour.reset_index()

train_knn = train_with_avg_hour.drop(['master_record_number'], axis=1).copy()
val_knn = val_with_avg_hour.drop(['master_record_number'], axis=1).copy()
test_knn = test_with_avg_hour.drop(['master_record_number'], axis=1).copy()

# In[19]:


cm = confusion_matrix(train_knn['time_window'], train_knn['most_common_time_window_{}_neigh'.format(k_value)])

cmd = ConfusionMatrixDisplay(cm, display_labels=np.unique(train_knn['time_window']))

fig, ax = plt.subplots(figsize=(10, 10))
cmd.plot(ax=ax, cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix for Training')
plt.xlabel('Predicted Time Window')
plt.ylabel('Actual Time Window')
plt.savefig(f'knn_cm_train_{k_value}_neigh_lat_long_only.png')

# In[21]:


cm = confusion_matrix(val_knn['time_window'], val_knn['most_common_time_window_{}_neigh'.format(k_value)])

cmd = ConfusionMatrixDisplay(cm, display_labels=np.unique(val_knn['time_window']))

fig, ax = plt.subplots(figsize=(10, 10))
cmd.plot(ax=ax, cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix for Validation')
plt.xlabel('Predicted Time Window')
plt.ylabel('Actual Time Window')
plt.savefig(f'knn_cm_val_{k_value}_neigh_lat_long_only.png')

# In[22]:


cm = confusion_matrix(test_knn['time_window'], test_knn['most_common_time_window_{}_neigh'.format(k_value)])

cmd = ConfusionMatrixDisplay(cm, display_labels=np.unique(test_knn['time_window']))

fig, ax = plt.subplots(figsize=(10, 10))
cmd.plot(ax=ax, cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix for Test')
plt.xlabel('Predicted Time Window')
plt.ylabel('Actual Time Window')
plt.savefig(f'knn_cm_test_{k_value}_neigh_lat_long_only.png')

# In[26]:


class_names = ['rush_morning', 'rush_evening', 'non_rush_day', 'non_rush_night']


# Function to calculate and print metrics
def print_metrics(y_true, y_pred, dataset_name):
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    print(f"\nClass-wise Metrics for {dataset_name} {k_value} nearest neighbors lat long only:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name} - Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1-Score: {f1_score[i]:.2f}")

    precision_macro, recall_macro, f1_score_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_micro, recall_micro, f1_score_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

    print("\nMacro Averages:")
    print(f"Precision: {precision_macro:.2f}, Recall: {recall_macro:.2f}, F1-Score: {f1_score_macro:.2f}")

    print("\nMicro Averages:")
    print(f"Precision: {precision_micro:.2f}, Recall: {recall_micro:.2f}, F1-Score: {f1_score_micro:.2f}")


# Validation Set Metrics and Confusion Matrix
y_true_val = val_knn['time_window']
y_pred_val = val_knn['most_common_time_window_{}_neigh'.format(k_value)]

print_metrics(y_true_val, y_pred_val, "Validation")

# Test Set Metrics and Confusion Matrix
y_true_test = test_knn['time_window']
y_pred_test = test_knn['most_common_time_window_{}_neigh'.format(k_value)]

print_metrics(y_true_test, y_pred_test, "Test")

# In[ ]:

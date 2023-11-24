import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from fancyimpute import MatrixFactorization


train_with_avg_hour = pd.read_csv('../03_Data_for_Modeling/train_matrix_factorization_new_time_window.csv')
val_with_avg_hour = pd.read_csv('../03_Data_for_Modeling/val_matrix_factorization_new_time_window.csv')

def categorize_time_window(hour):
    if 6 <= hour <= 9:
        return 0  # 'rush_morning'
    elif 15 <= hour <= 18:
        return 1  # 'rush_evening'
    elif 10 <= hour <= 14:
        return 2  # 'non_rush_day'
    else:
        return 3  # 'non_rush_night'

actual_hours_val = val_with_avg_hour['hour'].copy()
val_with_avg_hour['hour'] = np.nan
combined_df = pd.concat([train_with_avg_hour, val_with_avg_hour])
matrix_factorization = MatrixFactorization(learning_rate=0.0001, rank=4, max_iters=300, shrinkage_value=0.0001, min_value=0, max_value=23)
combined_imputed = matrix_factorization.fit_transform(combined_df)
combined_imputed_df = pd.DataFrame(combined_imputed, columns=combined_df.columns)
combined_imputed_df['hour'] = combined_imputed_df['hour'].round().astype(int)
combined_imputed_df['time_window'] = combined_imputed_df['hour'].apply(categorize_time_window)
num_val_rows = val_with_avg_hour.shape[0]
val_imputed = combined_imputed_df.iloc[-num_val_rows:]
actual_time_window_val = actual_hours_val.apply(categorize_time_window)
imputed_time_window_val = val_imputed['time_window']

y_true = actual_time_window_val
y_pred = imputed_time_window_val


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
class_names = ['rush_morning', 'rush_evening', 'non_rush_day', 'non_rush_night']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('cm_matrix_fact_0001lr_4rank_300maxiter_01shrinkage.png')


precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

print("Class-wise Metrics:")
for i, class_name in enumerate(class_names):
    print(f"{class_name} - Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1-Score: {f1_score[i]:.2f}")

precision_macro, recall_macro, f1_score_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
precision_micro, recall_micro, f1_score_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

print("\nMacro Averages:")
print(f"Precision: {precision_macro:.2f}, Recall: {recall_macro:.2f}, F1-Score: {f1_score_macro:.2f}")

print("\nMicro Averages:")
print(f"Precision: {precision_micro:.2f}, Recall: {recall_micro:.2f}, F1-Score: {f1_score_micro:.2f}")

accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
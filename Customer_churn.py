
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

possible_paths = [
    Path(__file__).resolve().parent / 'Customer-Churn.csv',
    Path.home() / 'Downloads' / 'Customer-Churn.csv',
]
DATA_PATH = next((path for path in possible_paths if path.exists()), None)

if DATA_PATH is None:
    raise FileNotFoundError(
        'Customer-Churn.csv was not found in the project folder or your Downloads folder.'
    )

dataset = pd.read_csv(DATA_PATH)

# Convert text values to numeric values where necessary.
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['TotalCharges'] = dataset['TotalCharges'].fillna(dataset['TotalCharges'].median())

print('Missing values:')
print(dataset.isnull().sum())
print('\nDataset summary:')
print(dataset.describe())

# ---- Matplotlib exploratory dashboard ----
fig, axes = plt.subplots(2, 2, figsize=(17, 11))
fig.suptitle('Customer Churn Overview', fontsize=16, fontweight='bold', y=0.98)

# Churn distribution
sns.countplot(
    x='Churn',
    hue='Churn',
    data=dataset,
    palette='coolwarm',
    legend=False,
    ax=axes[0, 0]
)
axes[0, 0].set_title('Customer Churn Distribution')
axes[0, 0].set_xlabel('Churn Status')
axes[0, 0].set_ylabel('Number of Customers')
axes[0, 0].grid(axis='y', alpha=0.25)

# Tenure distribution using a density curve
sns.kdeplot(
    data=dataset,
    x='tenure',
    fill=True,
    color='mediumseagreen',
    alpha=0.45,
    linewidth=2.5,
    ax=axes[0, 1]
)
axes[0, 1].set_title('Customer Tenure Distribution')
axes[0, 1].set_xlabel('Months with Company')
axes[0, 1].set_ylabel('Density')
axes[0, 1].grid(axis='y', alpha=0.25)

# Contract type and churn
contract_churn = pd.crosstab(dataset['Contract'], dataset['Churn'])
contract_churn.plot(kind='bar', ax=axes[1, 0], color=['steelblue', 'tomato'])
axes[1, 0].set_title('Churn by Contract Type')
axes[1, 0].set_xlabel('Contract Type')
axes[1, 0].set_ylabel('Number of Customers')
axes[1, 0].tick_params(axis='x', rotation=25, labelsize=9)
axes[1, 0].legend(title='Churn', loc='upper right')

# Monthly charges distribution
axes[1, 1].hist(dataset['MonthlyCharges'], bins=25, color='mediumpurple', edgecolor='black')
axes[1, 1].set_title('Monthly Charges Distribution')
axes[1, 1].set_xlabel('Monthly Charge ($)')
axes[1, 1].set_ylabel('Number of Customers')
axes[1, 1].grid(axis='y', alpha=0.25)

plt.tight_layout(pad=2.0, w_pad=3.2, h_pad=4.2, rect=[0, 0, 1, 0.95])
plt.subplots_adjust(hspace=0.38, wspace=0.28)
plt.show()
plt.close()

# ---- Model preparation ----
model_data = dataset.drop(columns=['customerID']).copy()
categorical_cols = model_data.select_dtypes(include=['object', 'str']).columns.drop('Churn')
label_encoders = {}

for column in categorical_cols:
    encoder = LabelEncoder()
    model_data[column] = encoder.fit_transform(model_data[column])
    label_encoders[column] = encoder

churn_encoder = LabelEncoder()
model_data['Churn'] = churn_encoder.fit_transform(model_data['Churn'])

X = model_data.drop(columns=['Churn'])
y = model_data['Churn']
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=0),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0)
}

predictions = {}
probabilities = {}
metrics = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions[name] = model.predict(X_test_scaled)
    probabilities[name] = model.predict_proba(X_test_scaled)[:, 1]
    metrics.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, predictions[name]),
        'Precision': precision_score(y_test, predictions[name]),
        'Recall': recall_score(y_test, predictions[name]),
        'F1': f1_score(y_test, predictions[name])
    })

metrics_df = pd.DataFrame(metrics).set_index('Model')
print('\nModel comparison:')
print(metrics_df.round(4))

# ---- Model performance dashboard ----
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold')

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
metric_colors = ['steelblue', 'seagreen', 'mediumpurple', 'tomato']
for index, metric_name in enumerate(metric_names):
    row, column = divmod(index, 2)
    axes[row, column].bar(metrics_df.index, metrics_df[metric_name],
                          color=metric_colors[index])
    axes[row, column].set_title(f'{metric_name} Comparison')
    axes[row, column].set_ylabel(metric_name)
    axes[row, column].set_ylim(0, 1)
    axes[row, column].tick_params(axis='x', rotation=0)
    for bar, value in zip(axes[row, column].patches, metrics_df[metric_name]):
        axes[row, column].text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                               f'{value:.2f}', ha='center', va='bottom')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
plt.close()

# Confusion matrix
random_forest = models['Random Forest']
y_pred = predictions['Random Forest']
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churn']
)
display.plot(cmap='Blues', values_format='d')
plt.title('Random Forest Confusion Matrix')
plt.show()
plt.close()

# ROC curves
plt.figure(figsize=(8, 6))
for name in models:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities[name])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate,
             label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.close()

# Random forest feature importance
importance = pd.Series(random_forest.feature_importances_, index=feature_names)
importance = importance.sort_values(ascending=True).tail(10)

plt.figure(figsize=(10, 6))
plt.barh(importance.index, importance.values, color='steelblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Random Forest Feature Importance')
plt.grid(axis='x', alpha=0.25)
plt.tight_layout()
plt.show()
plt.close()
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the data file
data_path = os.path.join(script_dir, 'data.csv')

# Load the breast cancer dataset
data = pd.read_csv(data_path)
data.drop(['Unnamed: 32','id'], axis=1, inplace=True)

# Map diagnosis to numerical values
data['diagnosis'] = data['diagnosis'].map({'B':0, 'M':1})

# Define features (X) and target (y)
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
class_names = ['Benign', 'Malignant']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SGDClassifier for logistic regression and train with loss tracking
model = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3)
loss_history = []
epochs = 100  # Number of passes over the training data

for i in range(epochs):
    model.partial_fit(X_train_scaled, y_train, classes=np.unique(y_train))
    
    # Calculate loss on the training set
    y_train_pred_proba = model.predict_proba(X_train_scaled)
    loss = log_loss(y_train, y_train_pred_proba)
    loss_history.append(loss)

# --- 1. ACCURACY SCORE ---
# Make predictions on the test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print("SKLEARN LOGISTIC REGRESSION RESULTS")
print('='*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"{'='*50}\n")


# --- 2. CONFUSION MATRIX ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix - Sklearn Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()


# --- 3. LOSS PLOT ---
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), loss_history, marker='o', linestyle='-')
plt.title('Training Loss Over Epochs - Sklearn')
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

"""
==================================================
SKLEARN LOGISTIC REGRESSION RESULTS
==================================================
Accuracy: 0.9737
==================================================
"""
# Accuracy: 0.9737
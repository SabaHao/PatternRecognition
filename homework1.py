# --- STEP 1 ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# --- STEP 1 --- 


# --- STEP 2 ---
from sklearn.metrics import roc_curve
# --- STEP 2 ---


# --- STEP 1 --- 
# Load Iris dataset
iris = datasets.load_iris()
# Select one feature, the sepal's length
X = iris.data[:, 0] 
y = (iris.target == 0).astype(int)

# Apply threshold to classify data
threshold = 5.0
y_pred = (X > threshold).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Classification based on threshold
y_train_pred = (X_train > threshold).astype(int)
y_test_pred = (X_test > threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
# --- STEP 1 --- 


# --- STEP 2 ---
# Predicted probabilities (just the feature values in this simple case)
probs = X

# Vary threshold and compute TPR and FPR
fpr, tpr, thresholds = roc_curve(y, probs)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')  # Diagonal line for reference
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Binary Classification')
plt.legend(loc="lower right")
plt.show()
# --- STEP 2 ---
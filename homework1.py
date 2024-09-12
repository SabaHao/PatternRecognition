import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# ----- LOAD THE DATA -----
iris = load_iris()
# Select the sepal length from the dataset
xAxis = iris.data[:, 0]
# Classify whether species is 'virginica'
y = (iris.target == 2).astype(int)

# ---- CLASSIFICATION -----
# Define a sample threshold for classification
threshold = 5
yPred = (xAxis > threshold).astype(int)

# Split the dataset into TRAIN and TEST
xTrain, xTest, yTrain, yTest = train_test_split(xAxis, y, test_size=0.3, random_state=42)

# ----- ACCURACY CALCULATION -----
accuracy = accuracy_score(yTest, (xTest > threshold).astype(int))
print(f'Accuracy: {accuracy:.2f}')

# Compute predicted probabilities
probabilities = xTest

# Vary the threshold to generate different classifications
thresholds = np.linspace(min(xAxis), max(xAxis), 100)
tpr = []
fpr = []

# ----- TRUE POSITIVE RATE && FALSE POSITIVE RATE
# Compute TPR and FPR for each threshold
for i in thresholds:
    yPred = (probabilities > i).astype(int)
    tp = np.sum((yPred == 1) & (yTest == 1))
    fn = np.sum((yPred == 0) & (yTest == 1))
    fp = np.sum((yPred == 1) & (yTest == 0))
    tn = np.sum((yPred == 0) & (yTest == 0))

    tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

# ----- PLOT -----
plt.figure()
plt.plot(fpr, tpr, color='green', label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')

# Labels
plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Show the plot
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.naive_bayes import GaussianNB

# ----- LOAD THE DATA -----
iris = load_iris()
# Select the sepal length from the dataset
X = iris.data[:, 0].reshape(-1, 1)  # Reshape to 2D array for the model
# Classify whether species is 'virginica'
y = (iris.target == 2).astype(int)

# ----- NAIVE BAYES CLASSIFIER -----
nb_model = GaussianNB()

# ----- CROSS-VALIDATION -----
# Perform cross-validation and get the accuracy for each fold
cv_scores = cross_val_score(nb_model, X, y, cv=5)  # 5-fold cross-validation
print(f'Cross-validated Accuracy Scores: {cv_scores}')
print(f'Average Cross-validated Accuracy: {np.mean(cv_scores):.2f}')

# ----- SPLIT THE DATA INTO TRAIN AND TEST -----
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model on the training data
nb_model.fit(xTrain, yTrain)

# ----- ACCURACY CALCULATION -----
yPred = nb_model.predict(xTest)
accuracy = accuracy_score(yTest, yPred)
print(f'Accuracy on Test Set: {accuracy:.2f}')

# Compute predicted probabilities
yProba = nb_model.predict_proba(xTest)[:, 1]  # Get probabilities for the positive class

# ----- ROC CURVE -----
# Vary the threshold to generate different classifications
thresholds = np.linspace(0, 1, 100)
tpr = []
fpr = []

# ----- TRUE POSITIVE RATE && FALSE POSITIVE RATE
# Compute TPR and FPR for each threshold
for i in thresholds:
    yPred = (yProba > i).astype(int)
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



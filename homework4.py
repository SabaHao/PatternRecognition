import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Binarize the output labels for multiclass ROC curve
y_bin = label_binarize(y, classes=[0, 1, 2])

# classifiers
lda = LinearDiscriminantAnalysis()
logreg = LogisticRegression(max_iter=1000, random_state=42)
nb = GaussianNB()

# Train classifiers
lda.fit(X, y)
logreg.fit(X, y)
nb.fit(X, y)

# Predict for all classifiers
y_pred_lda = lda.predict(X)
y_pred_logreg = logreg.predict(X)
y_pred_nb = nb.predict(X)

# Probabilities for ROC curve
y_probs_lda = lda.predict_proba(X)
y_probs_logreg = logreg.predict_proba(X)
y_probs_nb = nb.predict_proba(X)

# F1 Scores and Accuracy
f1_lda = f1_score(y, y_pred_lda, average='weighted')
f1_logreg = f1_score(y, y_pred_logreg, average='weighted')
f1_nb = f1_score(y, y_pred_nb, average='weighted')

acc_lda = accuracy_score(y, y_pred_lda)
acc_logreg = accuracy_score(y, y_pred_logreg)
acc_nb = accuracy_score(y, y_pred_nb)

print(f"LDA F1 Score: {f1_lda:.2f}, Accuracy: {acc_lda:.2f}")
print(f"Logistic Regression F1 Score: {f1_logreg:.2f}, Accuracy: {acc_logreg:.2f}")
print(f"Naive Bayes F1 Score: {f1_nb:.2f}, Accuracy: {acc_nb:.2f}")

# ROC curves
fpr_lda, tpr_lda, _ = roc_curve(y_bin.ravel(), y_probs_lda.ravel())
fpr_logreg, tpr_logreg, _ = roc_curve(y_bin.ravel(), y_probs_logreg.ravel())
fpr_nb, tpr_nb, _ = roc_curve(y_bin.ravel(), y_probs_nb.ravel())

# AUC scores
roc_auc_lda = auc(fpr_lda, tpr_lda)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)
roc_auc_nb = auc(fpr_nb, tpr_nb)

# Plot ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_lda, tpr_lda, label=f'LDA (AUC = {roc_auc_lda:.2f})', color='blue')
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {roc_auc_logreg:.2f})', color='green')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})', color='red')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for LDA, Logistic Regression, and Naive Bayes')
plt.legend(loc='lower right')
plt.show()

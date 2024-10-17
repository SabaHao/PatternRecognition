import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, roc_auc_score


# Load Iris dataset
iris = datasets.load_iris()

# Here in homework 2 I will take every feature, not only the petal length
X = iris.data
y = (iris.target == 0).astype(int)  # Binary classification: 'setosa' vs. non-setosa

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 1: Train 
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Step 2: Predict probabilities 
y_probs = nb_model.predict_proba(X_test)[:, 1]  

# Step 3: ompute ROC curve and AUC score for the test set
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)


# Here is the optional part about the cross validation
cv = StratifiedKFold(n_splits=5)
y_probs_cv = cross_val_predict(nb_model, X, y, cv=cv, method='predict_proba')[:, 1]

# Ste 5: ROC curve and AUC for cross-validated results
fpr_cv, tpr_cv, _ = roc_curve(y, y_probs_cv)
roc_auc_cv = roc_auc_score(y, y_probs_cv)

# Plot both ROC curves on the same plot
plt.figure()
plt.plot(fpr, tpr, label=f'Test Set ROC (AUC = {roc_auc:.2f})', color='blue')
plt.plot(fpr_cv, tpr_cv, label=f'Cross-Validated ROC (AUC = {roc_auc_cv:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')

# Labels and title
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves for Naive Bayes Classifier')
plt.legend(loc="lower right")

# Show plot
plt.show()

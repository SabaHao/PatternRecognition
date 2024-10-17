import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ----- LOAD THE DATA -----
iris = load_iris()
# Use all four features (sepal length, sepal width, petal length, petal width)
X = iris.data
# Binary classification for 'virginica' species
y = (iris.target == 2).astype(int)

# ----- SPLIT THE DATA -----
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- MODEL INSTANTIATION -----
# Naive Bayes Classifier
nb_model = GaussianNB()
# Logistic Regression Classifier
logreg_model = LogisticRegression()
# Linear Discriminant Analysis
lda_model = LinearDiscriminantAnalysis()

# ----- TRAINING -----
nb_model.fit(xTrain, yTrain)
logreg_model.fit(xTrain, yTrain)
lda_model.fit(xTrain, yTrain)

# ----- PREDICTION -----
yPred_nb = nb_model.predict(xTest)
yPred_logreg = logreg_model.predict(xTest)
yPred_lda = lda_model.predict(xTest)

# ----- PROBABILITY PREDICTIONS FOR ROC CURVE -----
yProba_nb = nb_model.predict_proba(xTest)[:, 1]
yProba_logreg = logreg_model.predict_proba(xTest)[:, 1]
yProba_lda = lda_model.predict_proba(xTest)[:, 1]

# ----- PERFORMANCE METRICS -----
# Accuracy
accuracy_nb = accuracy_score(yTest, yPred_nb)
accuracy_logreg = accuracy_score(yTest, yPred_logreg)
accuracy_lda = accuracy_score(yTest, yPred_lda)

# F1 Score
f1_nb = f1_score(yTest, yPred_nb)
f1_logreg = f1_score(yTest, yPred_logreg)
f1_lda = f1_score(yTest, yPred_lda)

# ----- ROC CURVE -----
# Compute ROC curve and AUC for each model
fpr_nb, tpr_nb, _ = roc_curve(yTest, yProba_nb)
fpr_logreg, tpr_logreg, _ = roc_curve(yTest, yProba_logreg)
fpr_lda, tpr_lda, _ = roc_curve(yTest, yProba_lda)

auc_nb = auc(fpr_nb, tpr_nb)
auc_logreg = auc(fpr_logreg, tpr_logreg)
auc_lda = auc(fpr_lda, tpr_lda)

# ----- DISPLAY PERFORMANCE -----
print("Naive Bayes Classifier: Accuracy = {:.2f}, F1 Score = {:.2f}, AUC = {:.2f}".format(accuracy_nb, f1_nb, auc_nb))
print("Logistic Regression: Accuracy = {:.2f}, F1 Score = {:.2f}, AUC = {:.2f}".format(accuracy_logreg, f1_logreg, auc_logreg))
print("Linear Discriminant Analysis: Accuracy = {:.2f}, F1 Score = {:.2f}, AUC = {:.2f}".format(accuracy_lda, f1_lda, auc_lda))

# ----- PLOT ROC CURVES -----
plt.figure()
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes (AUC = %0.2f)' % auc_nb, color='blue')
plt.plot(fpr_logreg, tpr_logreg, label='Logistic Regression (AUC = %0.2f)' % auc_logreg, color='green')
plt.plot(fpr_lda, tpr_lda, label='LDA (AUC = %0.2f)' % auc_lda, color='purple')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')

# Labels and Legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid()
plt.show()

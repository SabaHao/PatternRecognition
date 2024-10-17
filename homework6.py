import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the diabetes dataset I used the dataset from wechat not from the sktlearn. 
data = pd.read_csv('diabetes.csv')

X = data.drop('Outcome', axis=1)  
y = data['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the SVM model
svm = SVC()

# Define the hyperparameter grid
# Reduced hyperparameter grid example
param_grid = {
    'C': [0.1, 1],  # Fewer candidates
    'gamma': [0.01, 0.1],
    'kernel': ['linear', 'rbf']  # Fewer kernel types
}

grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=1)  # Using 3 folds


# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, 
                           scoring='accuracy', cv=5, verbose=1)

# Fit the model with GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the SVM model with the best parameters
best_svm = SVC(**best_params)
best_svm.fit(X_train, y_train)

# Evaluate the model on the test set
test_accuracy = best_svm.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")


# Predictions
y_pred = best_svm.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

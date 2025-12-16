from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# X -> feature matrix (numpy array)
# y -> target labels (numpy array)


model = LogisticRegression(max_iter=1000)

initial_window = 200   # starting training size
predictions = []
actuals = []

for i in range(initial_window, len(X)):
    X_train = X[:i]
    y_train = y[:i]

    X_test = X[i].reshape(1, -1)
    y_test = y[i]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    predictions.append(y_pred[0])
    actuals.append(y_test)

# Evaluation
accuracy = accuracy_score(actuals, predictions)
cm = confusion_matrix(actuals, predictions)

print("Walk-Forward Validation Accuracy:", round(accuracy, 4))
print("Confusion Matrix:")
print(cm)

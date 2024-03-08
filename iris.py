import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load the Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit Logistic Regression to the Training set
classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
classifier.fit(X_train, y_train)

# Predict the Test set results and probabilities
y_pred = classifier.predict(X_test)
probs_y = np.round(classifier.predict_proba(X_test), 2)

# Print results
print("{:<10} | {:<10} | {:<10} | {:<13} | {:<5}".format("y_test", "y_pred", "Setosa(%)", "Versicolor(%)", "Virginica(%)\n"))
print("-"*65)
for x, y, a, b, c in zip(y_test, y_pred, probs_y[:, 0], probs_y[:, 1], probs_y[:, 2]):
    print("{:<10} | {:<10} | {:<10} | {:<13} | {:<10}".format(x, y, a, b, c))
print("-"*65)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print confusion matrix as text
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, annot_kws={"size": 12}, fmt='d', cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv('advertising.csv')

# Data Exploration
dataset.head(10)
dataset.shape
dataset.isna().sum()
dataset.duplicated().any()

# Data Visualization
fig, axs = plt.subplots(3, figsize=(5, 5))
for i, col in enumerate(['TV', 'Newspaper', 'Radio']):
    sns.boxplot(dataset[col], ax=axs[i])
plt.tight_layout()
plt.show()

sns.distplot(dataset['Sales'])
plt.show()

sns.pairplot(dataset, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

sns.heatmap(dataset.corr(), annot=True)
plt.show()

# Model Training
x = dataset[['TV']]
y = dataset['Sales']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

slr = LinearRegression()
slr.fit(x_train, y_train)

# Model Evaluation
print('Intercept:', slr.intercept_)
print('Coefficient:', slr.coef_)
print('Regression Equation: Sales = {:.3f} + {:.3f} * TV'.format(slr.intercept_, slr.coef_[0]))

plt.scatter(x_train, y_train)
plt.plot(x_train, slr.predict(x_train), 'r')
plt.show()

y_pred_slr = slr.predict(x_test)
print("Prediction for test set:", y_pred_slr)

print('R squared value of the model: {:.2f}'.format(slr.score(x, y) * 100))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv("Advertising.csv")
dataset.head()

dataset.shape

dataset.isna().sum()

dataset.duplicated().any()

fig, axs=plt.subplots(3,figsize=(5,5))
plt1=sns.boxplot(dataset['TV'],ax=axs[0])
plt2=sns.boxplot(dataset['Newspaper'],ax=axs[1])
plt3=sns.boxplot(dataset['Radio'],ax=axs[2])
plt.tight_layout()

sns.distplot(dataset['Sales']);

sns.pairplot(dataset,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',height=4,aspect=1,kind='scatter')
plt.show()

sns.heatmap(dataset.corr(),annot=True)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
x=dataset[['TV']]
y=dataset['Sales']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
slr=LinearRegression()
slr.fit(x_train,y_train)
print('Intercept: ',slr.intercept_)
print('Coefficient: ',slr.coef_)

print('Regression Equation sales=6.948+0.054*TV')

plt.scatter(x_train,y_train)
plt.plot(x_train,6.948+0.054*x_train,'r')
plt.show()

y_pred_slr=slr.predict(x_test)
x_pred_slr=slr.predict(x_train)
print("predication for test set:{}".format(y_pred_slr))

slr_diff=pd.DataFrame({'Actual value':y_test,'Predicated Value':y_pred_slr})
slr_diff

meanAbErr=metrics.mean_absolute_error(y_test,y_pred_slr)
meanSqErr=metrics.mean_squared_error(y_test,y_pred_slr)
rootMeanSqErr=np.sqrt(metrics.mean_squared_error(y_test,y_pred_slr))
print('mean absolute error',meanAbErr)
print('mean square error',meanSqErr)
print('root mean square error',rootMeanSqErr)
slr.predict([[56]])

from sklearn.metrics import accuracy_score
print('R squred value of the model:{:.2f}'.format(slr.score(x,y)*100))

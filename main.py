import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

titanic_data = pd.read_csv('titanic.csv', encoding='windows-1251', sep=',')
titanic_data.isnull().sum() #определяем, сколько пропущено данных в переменных, возможно что-то стоит исключить

X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data.Survived

X = pd.get_dummies(X)
X.Age.median()
X = X.fillna({'Age': X.Age.median()})

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_test, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf.score(X, y)
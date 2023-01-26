import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score

titanic_data = pd.read_csv('titanic.csv', encoding='windows-1251', sep=',')
titanic_data.isnull().sum() #определяем, сколько пропущено данных в переменных, возможно что-то стоит исключить

X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1) #удаляем ненужные столбцы и тот, который будем предсказывать
y = titanic_data.Survived

X = pd.get_dummies(X)
X.Age.median()
X = X.fillna({'Age': X.Age.median()})

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf.score(X, y)

#посмотрим, как влияет глубина выборки на точность предсказывания значений деревом решений - переобучили модель
scores_data = pd.DataFrame()
max_depth_values = range(1, 100)
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()

    temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                    'train_score': [train_score],
                                    'test_score': [test_score],
                                    'cross_val_score': [mean_cross_val_score]})

    scores_data = pd.concat([scores_data, temp_score_data])

scores_data_long = pd.melt(scores_data, id_vars=['max_depth'],
                           value_vars=['train_score', 'test_score', 'cross_val_score'],
                           var_name='set_type',
                           value_name='score')

from sklearn.model_selection import GridSearchCV
clf = tree.DecisionTreeClassifier()
parametrs = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}
grid_search_cv_clf = GridSearchCV(clf, parametrs, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
grid_search_cv_clf.best_params_
best_clf = grid_search_cv_clf.best_estimator_

from sklearn.metrics import precision_score, recall_score
y_pred = best_clf.predict(X_test)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
y_predicted_prob = best_clf.predict_proba(X_test)
pd.Series(y_predicted_prob[:, 1]).hist()
y_pred = np.where(y_predicted_prob[:, 1] > 0.8, 1, 0)

#строим ROC кривую



from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import pandas as pd
import numpy as np

df = pd.read_csv("../../stock_avg.csv")
Y = df['label']
X = df.drop(['SecuCode','SecuAbbr','TradingDay','label'], axis=1)

X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,10,5,5))

kf = KFold(n_splits=5, shuffle=True)

count = 1
for train, test in kf.split(X):
    x_train, x_test = X.iloc[train], X.iloc[test]
    y_train, y_test = Y.iloc[train], Y.iloc[test]

    mlp.fit(x_train, y_train)
    result = mlp.predict(x_test)

    precision = precision_score(y_test, result)
    recall = recall_score(y_test, result)
    accuracy = accuracy_score(y_test, result)
    f1 = f1_score(y_test, result)

    print("\t Accuracy :%f, Precision :%f, Recall :%f, F1 :%f" % (accuracy, precision, recall, f1))
    s = joblib.dump(mlp, '../store/task1-mlp1-accuracy' + str(accuracy))

    count += 1




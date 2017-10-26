import pandas as pd
from sklearn.externals import joblib
import base
# CSV file columns name

df = base.load_strategy1_data()

mlp = joblib.load('./task1/store/task1-mlp1-accuracy0.736578983101')
q = df.groupby(['TradingDay', 'SecuCode','SecuAbbr']).apply(lambda x: mlp.predict_proba(x.iloc[:, 3:9])[0,1]).reset_index(name='Q')

sorted_10 = q.set_index(['SecuAbbr']).groupby(['TradingDay'])['Q'].apply(lambda x: x.sort_values(ascending=False).head(10))

sorted_10.to_csv("./sorted_10.csv")

print(sorted_10)
import pandas as pd
from sklearn.externals import joblib

df = pd.read_csv("./stock_avg.csv")

q = df.groupby(['TradingDay', 'SecuCode']).apply(lambda x: mlp.predict_proba(x)).reset_index(name='Q')

print(q)

mlp = joblib.load('./store/task1-mlp1-accuracs y0.999725963141')
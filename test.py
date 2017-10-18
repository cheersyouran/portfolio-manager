import pandas as pd

df = pd.read_csv("~/Desktop/Snowball/quote.csv")

a = df.groupby('SecuAbbr').nunique()
print(a)

stock1 = df[df['SecuAbbr']=='平安银行'].drop(['SecuCode','SecuAbbr'], 1)

x1=stock1[["Close","High","Low"]]

X =[x1,x1]
# format='%Y-%m-%d %H:%M:%S'

stock1["TradingDay"] = stock1["TradingDay"].apply(lambda x : pd.to_datetime(x).date());

import pandas as pd

ALL_HISTORY=719
TIME_SPACE=50
HISTORY=ALL_HISTORY-TIME_SPACE-1
STOCK_NUM=1

QUTOE_PATH="~/Desktop/Snowball/quote.csv"
RECORDS_PATH="~/Desktop/Snowball/records.csv"
NAV_PATH="~/Desktop/Snowball/nav.csv"

MODEL1_PATH="./store/model1.ckpt"

def load_records_csv():
    print('load records data....')
    df = pd.read_csv(RECORDS_PATH, parse_dates=['Updated'])
    df["Updated"] = df["Updated"].apply(lambda x: pd.to_datetime(x))
    return df.drop(df.columns[0], axis=1)

def load_quote_csv():
    print('load quote data....')
    df = pd.read_csv(QUTOE_PATH, parse_dates=['TradingDay'])
    df["TradingDay"] = df["TradingDay"].apply(lambda x: pd.to_datetime(x))
    return df.drop(df.columns[0], axis=1)

def load_nav_csv():
    print('load nav data....')
    df = pd.read_csv(NAV_PATH, parse_dates=['NavDate'])
    df["NavDate"] = df["NavDate"].apply(lambda x: pd.to_datetime(x))
    return df.drop(df.columns[0], axis=1)


# ['SecuCode', 'SecuAbbr', 'TradingDay', '2_avg_diff_ratio',
# '3_avg_diff_ratio', '4_avg_diff_ratio', '5_avg_diff_ratio',
# '6_avg_diff_ratio', '7_avg_diff_ratio', 'label']
def load_strategy1_data():
    print('load strategy1 data....')
    return pd.read_csv("/Users/Youran/Projects/PortfolioManagement/stock_avg.csv", parse_dates=['TradingDay'])
import pandas as pd

ALL_HISTORY=719
TIME_SPACE=50
HISTORY=ALL_HISTORY-TIME_SPACE-1
STOCK_NUM=1

PROJ_PATH='~/Projects/PortfolioManagement'

QUTOE_PATH=PROJ_PATH+"/data/quote.csv"
RECORDS_PATH=PROJ_PATH+"/data/records.csv"
NAV_PATH=PROJ_PATH+"/data/nav.csv"
INDUTRY_PATH=PROJ_PATH+'/data/industry.csv'

S1_PATH=PROJ_PATH+'/stock_avg.csv'
MODEL1_PATH=PROJ_PATH+"/store/model1.ckpt"
STATE_SPACE=PROJ_PATH+'/State_Space/rolling_data1.csv'

def load_records_csv():
    print('load records data....')
    df = pd.read_csv(RECORDS_PATH, parse_dates=['Updated'])
    df["Updated"] = df["Updated"].apply(lambda x: pd.to_datetime(x))
    return df

def load_quote_csv():
    print('load quote data....')
    df = pd.read_csv(QUTOE_PATH, parse_dates=['TradingDay'])
    df["TradingDay"] = df["TradingDay"].apply(lambda x: pd.to_datetime(x))
    return df

def load_nav_csv():
    print('load nav data....')
    df = pd.read_csv(NAV_PATH, parse_dates=['NavDate'])
    df["NavDate"] = df["NavDate"].apply(lambda x: pd.to_datetime(x))
    return df

def load_industry_csv():
    print('load nav data....')
    df = pd.read_csv(INDUTRY_PATH)
    return df

def load_states_csv():
    print('load state space data....')
    df = pd.read_csv(STATE_SPACE)
    return df

# ['SecuCode', 'SecuAbbr', 'TradingDay', '2_avg_diff_ratio',
# '3_avg_diff_ratio', '4_avg_diff_ratio', '5_avg_diff_ratio',
# '6_avg_diff_ratio', '7_avg_diff_ratio', 'label']
def load_strategy1_data():
    print('load strategy1 data....')
    return pd.read_csv(S1_PATH, parse_dates=['TradingDay'])
#encoding:utf-8
import pandas as pd

PROJ_PATH='/Users/wangchengming/Documents/5001Project/Snowball/RL'
DATE='2017-04-05'

QUTOE_PATH=PROJ_PATH+"/data/quote.csv"
RECORDS_PATH=PROJ_PATH+"/data/records.csv"
NAV_PATH=PROJ_PATH+"/data/nav.csv"
INDUTRY_PATH=PROJ_PATH+'/data/industry.csv'
INDUSTRYQUOTE_PATH=PROJ_PATH+'/data/industry_quote.xlsx'
IR_WEEK_PATH=PROJ_PATH+'/State_Space/IR_rank_week.csv'
CERTAIN_STATE=PROJ_PATH+'/State_Space/银行_MACD_5_10_BOLL.csv'
TRADING_DAY=PROJ_PATH+'/data/tradingday.csv'

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
    print('load industry data....')
    df = pd.read_csv(INDUTRY_PATH)
    return df

def load_industryquote_xlsx():
    print('load industry quote data...')
    df = pd.read_excel(INDUSTRYQUOTE_PATH)
    df.columns = df.iloc[0, :].apply(lambda x: x[:-4]).values
    df = df.iloc[2:]
    df.index.names = ['TradingDay']
    df.reset_index(inplace=True)
    return df

def load_irweek_csv():
    print('load information ratio rank data...')
    df = pd.read_csv(IR_WEEK_PATH, index_col='PortCode')
    return df

def load_certain_states_csv():
    print('load ', CERTAIN_STATE, 'data...')
    df = pd.read_csv(CERTAIN_STATE)
    return df

def load_trading_day_csv():
    print('load trading day data....')
    return pd.read_csv(TRADING_DAY,  parse_dates=['TradingDate'])
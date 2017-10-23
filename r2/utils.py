import pandas as pd

from r2 import config

TIME_SPACE = config.TIME_SPACE
STOCK_NUM = config.STOCK_NUM
HISTORY = config.HISTORY
ALL_HISTORY = config.ALL_HISTORY
PATH = config.PATH

column = ['SecuCode','SecuAbbr','TradingDay','Close','High','Low','Avg','ChangePCT','TurnoverVolume','TurnoverValue']
ex_column = []

def load_data(path):
    df = pd.read_csv(path, parse_dates=['TradingDay'])
    return df

def with_nday_avg(df, n_day):
    # df1 = df.groupby('SecuAbbr').apply(lambda x: x.set_index('TradingDay'))
    # avg = df1.groupby(level=0)['Close']\
    #     .apply(lambda x: x.rolling(min_periods=n_day, window=n_day).mean())\
    #     .reset_index(name=str(n_day)+'_Avg')

    # gb = df.groupby("SecuAbbr")
    # for x in gb.groups:
    #     stock=gb.get_group(x)

    #NOTE:rolling().apply()
    col = str(n_day)+'_avg'
    avg = df.set_index(['SecuCode', 'TradingDay']).groupby('SecuCode')['Close']\
        .apply(lambda x: x.rolling(min_periods=n_day, window=n_day).mean())\
        .reset_index(name=col)
    merged = pd.merge(df, avg, on=['SecuCode', 'TradingDay'], how='left')

    ex_column.append(col)
    return merged


def with_nday_reward_ratio(df, n_day):
    col = str(n_day)+'_reward_ratio'
    ratio = df.set_index(['SecuCode','TradingDay']).groupby('SecuCode')['Close'] \
        .apply(lambda x: x/x.shift(n_day-1))\
        .reset_index(name=col)
    merged = pd.merge(df, ratio, on=['SecuCode', 'TradingDay'], how='left')

    # ex_column.append(col)
    return merged

def label_by_reward_ratio(df, n_day, threshold = 0):

    col = str(n_day)+'_reward_ratio'
    df.loc[df[col] <= 1 - threshold, 'label'] = 0
    df.loc[df[col] >= 1 + threshold, 'label'] = 1
    df = df.drop(df[(df[col] < 1+threshold) & (df[col] > 1-threshold)].index)

    # ex_column.append(col)
    return df

def with_nday_ratio_difference(df, n_day):
    col = str(n_day) + '_avg_diff_ratio'
    avg = str(n_day) + '_avg'
    df[col] = (df[avg] - df['Close']) / df[avg]

    # ex_column.append(col)
    return df


def process_data():
    df = load_data(PATH)
    df = with_nday_avg(df, 2)
    df = with_nday_avg(df, 3)
    df = with_nday_avg(df, 4)
    df = with_nday_avg(df, 5)
    df = with_nday_avg(df, 6)
    df = with_nday_avg(df, 7)

    df = with_nday_ratio_difference(df, 2)
    df = with_nday_ratio_difference(df, 3)
    df = with_nday_ratio_difference(df, 4)
    df = with_nday_ratio_difference(df, 5)
    df = with_nday_ratio_difference(df, 6)
    df = with_nday_ratio_difference(df, 7)

    df = with_nday_reward_ratio(df, 15)
    df = label_by_reward_ratio(df, 15)

    df = df.drop(column[3:] + ex_column, axis=1)
    df = df.dropna(axis=0, how='any')

    df.to_csv("../stock_avg.csv", index=False)

process_data()

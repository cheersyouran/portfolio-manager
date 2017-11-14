import pandas as pd
import base

TIME_SPACE = base.TIME_SPACE
STOCK_NUM = base.STOCK_NUM
HISTORY = base.HISTORY
ALL_HISTORY = base.ALL_HISTORY

column = ['SecuCode', 'SecuAbbr', 'TradingDay',
          'Close', 'High', 'Low', 'Avg', 'ChangePCT',
          'TurnoverVolume', 'TurnoverValue']
ex_column = []

#计算n日均价格
def with_nday_avg(df, n_day):

    # gb = df.groupby("SecuAbbr")
    # for x in gb.groups:
    #     stock=gb.get_group(x)

    col = str(n_day)+'_avg'
    avg = df.set_index(['SecuCode', 'TradingDay']).groupby('SecuCode')['Close']\
        .apply(lambda x: x.rolling(min_periods=n_day, window=n_day).mean())\
        .reset_index(name=col) #NOTE:rolling().apply()
    merged = pd.merge(df, avg, on=['SecuCode', 'TradingDay'], how='left')

    ex_column.append(col)
    return merged

#计算n日回报率
def with_nday_reward_ratio(df, n_day):
    col = str(n_day)+'_reward_ratio'
    ratio = df.set_index(['SecuCode','TradingDay']).groupby('SecuCode')['Close'] \
        .apply(lambda x: x/x.shift(n_day-1))\
        .reset_index(name=col)
    merged = pd.merge(df, ratio, on=['SecuCode', 'TradingDay'], how='left')

    ex_column.append(col)
    return merged

#用n日的回报率和threshold做0or1监督标注
def label_by_reward_ratio(df, n_day, threshold = 0):

    col = str(n_day)+'_reward_ratio'
    df.loc[df[col] <= 1 - threshold, 'label'] = 0
    df.loc[df[col] >= 1 + threshold, 'label'] = 1
    df = df.drop(df[(df[col] < 1+threshold) & (df[col] > 1-threshold)].index)

    # ex_column.append(col)
    return df

#计算（n日均价格 - 当日收盘价）
def with_nday_ratio_difference(df, n_day):
    col = str(n_day) + '_avg_diff_ratio'
    avg = str(n_day) + '_avg'
    df[col] = (df[avg] - df['Close']) / df[avg]

    # ex_column.append(col)
    return df

#生成数据
def process_data():
    df = base.load_quote_csv()
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

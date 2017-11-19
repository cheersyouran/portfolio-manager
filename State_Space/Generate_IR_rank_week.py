
import numpy as np
import base
import pandas as pd
from datetime import datetime

industry = base.load_industry_csv()


def year_week(x):
    year = str(pd.Timestamp(x).isocalendar()[0])
    week = str(pd.Timestamp(x).isocalendar()[1])
    if len(week)==1:
        week = '0' + week
    return year + '-' + week


# Demo:
#       generate_IR_rank(df_records, df_nav, df_industryquote, df_quote, save_path='./data')
#       Given records, nav, industry quote and quote data, generate IR rank and save it to data folder.

def generate_IR_rank(df_records, df_nav, df_industryquote, df_quote, save_path=None):

    # get year-week info
    nav_copy = df_nav.copy()
    nav_copy.NavDate = nav_copy.NavDate.apply(year_week)
    # weekly nav ratio
    nav_week_ratio = nav_copy.groupby(['PortCode', 'NavDate']).apply(
        lambda x: (x.Nav.values[-1] - x.Nav.values[0])/x.Nav.values[0])
    nav_week_ratio = pd.DataFrame(nav_week_ratio)
    nav_week_ratio.columns = ['Ratio']

    # For each portcode and every week, get the secucodes.
    records_week = df_records.copy()
    records_week.Updated = records_week.Updated.apply(year_week)
    records_week = records_week.groupby(['PortCode', 'Updated']).apply(
        lambda x: x.SecuCode)
    records_week = pd.DataFrame(records_week)

    # get weekly ratio for every stock
    quote_copy = df_quote.copy()
    quote_copy.TradingDay = quote_copy.TradingDay.apply(year_week)
    quote_copy = quote_copy.groupby(['SecuCode', 'TradingDay']).apply(
        lambda x: (x.Close.values[-1] - x.Close.values[0]) / x.Close.values[0])
    quote_oc = pd.DataFrame(quote_copy)
    quote_oc.columns = ['Ratio']

    records_temp = records_week.reset_index()
    codes = set(df_records.SecuCode.unique()) - set(df_quote.SecuCode.unique())
    codes = list(codes.intersection(set(industry.SecuCode.unique())))
    temp = industry[industry.SecuCode.isin(codes)].set_index(['SecuCode'])
    dicts = temp.to_dict()['FirstIndustryName']

    # If a security is in the quote data, no changes.
    # If it's not in the quote data but in the industry data, then replace it with industry data.
    # If neither, return NaN. There's nothing we can do.
    records_temp['Secu_ind'] = records_temp.SecuCode.map(dicts)
    records_temp.Secu_ind.fillna(records_temp.SecuCode, inplace=True)
    records_temp.drop(['SecuCode', 'level_2'], axis=1, inplace=True)
    records_temp.columns = ['PortCode', 'Updated', 'SecuCode']

    # get industry data. Basically the same operation as above.
    industry_copy = df_industryquote.copy()
    industry_copy.TradingDay = df_industryquote.TradingDay.apply(year_week)
    industry_copy = industry_copy.set_index('TradingDay')
    ind = industry_copy.index
    industry_copy.columns.names = ['Industry']
    industry_copy = industry_copy.melt()
    industry_copy.index = ind.tolist() * 29
    industry_copy = industry_copy.reset_index()
    industry_copy.columns = ['TradingDay', 'Industry', 'Close']

    industry_week_ratio = industry_copy.groupby(['Industry', 'TradingDay']).apply(
        lambda x: (x.Close.values[-1] - x.Close.values[0]) / x.Close.values[0])
    industry_week_ratio = pd.DataFrame(industry_week_ratio)
    industry_week_ratio.columns = ['Ratio']

    # combine industry data with quote data
    industry_week_ratio1 = industry_week_ratio.reset_index()
    industry_week_ratio1.columns = ['SecuCode', 'TradingDay', 'Ratio']
    quote_temp = quote_oc.reset_index()
    quote_temp = pd.concat([quote_temp, industry_week_ratio1])

    market = records_temp.groupby(['PortCode']).apply(lambda x: quote_temp[quote_temp.SecuCode.isin(
                x.SecuCode.unique())]).reset_index()
    market_week_ratio = market.groupby(['PortCode', 'TradingDay']).apply(lambda x: x.Ratio.mean())
    market_week_ratio = pd.DataFrame(market_week_ratio).reset_index()

    # merge market_week_ratio to nav datset in order to compare with nav_week_ratio
    nav_temp = nav_copy.groupby(['PortCode', 'NavDate']).apply(lambda x: x.iloc[0])
    nav_temp = nav_temp.drop(['NavDate', 'PortCode'], axis=1).reset_index()
    market_week_ratio = nav_temp.merge(market_week_ratio, how='left', left_on=['PortCode', 'NavDate'], 
                                                      right_on=['PortCode', 'TradingDay'])
    market_week_ratio.drop(['TradingDay', 'ID', 'Nav'], axis=1, inplace=True)
    market_week_ratio.columns = ['PortCode', 'NavDate', 'Ratio']

    nav_week_ratio_reset = nav_week_ratio.reset_index()
    full = market_week_ratio.merge(nav_week_ratio_reset, 
                                     left_on=['PortCode', 'NavDate'], right_on=['PortCode', 'NavDate'])
    full['excess'] = full.Ratio_y - full.Ratio_x
    result = pd.DataFrame(full.groupby(['PortCode']).apply(lambda x: x.excess.mean() / x.excess.std()))
    result['num_week'] = full.groupby(['PortCode']).apply(lambda x: len(x))
    result['mean'] = full.groupby(['PortCode']).apply(lambda x: x.excess.mean())
    result['std'] = full.groupby(['PortCode']).apply(lambda x: x.excess.std())
    result.columns = ['IR', 'num_week', 'mean', 'std']

    IR_rank = result[~pd.isnull(result.IR)].sort_values(['IR'])

    if save_path is None:
        return IR_rank
    else:
        IR_rank.to_csv(save_path + '/IR_rank_week.csv')
        print('Saved IR_rank_week to', save_path)
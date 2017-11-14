
import numpy as np
import pandas as pd
import base

quote = base.load_quote_csv()
industry = base.load_industry_csv()
records = base.load_records_csv()
nav = base.load_nav_csv()
industry_quote = base.load_industryquote_xlsx()


# get year-month info
nav_copy = nav.copy()
nav_copy.NavDate = nav_copy.NavDate.apply(lambda x: str(x)[:7])
# monthly nav ratio
nav_month_ratio = nav_copy.groupby(['PortCode', 'NavDate']).apply(
    lambda x: (x.Nav.values[-1] - x.Nav.values[0])/x.Nav.values[0])
nav_month_ratio = pd.DataFrame(nav_month_ratio)
nav_month_ratio.columns = ['Ratio']


# For each portcode and every month, get the secucodes.
records_month = records.copy()
records_month.Updated = records_month.Updated.apply(lambda x: str(x)[:7])
records_month = records_month.groupby(['PortCode', 'Updated']).apply(
    lambda x: x.SecuCode)
records_month = pd.DataFrame(records_month)


# get monthly ratio
quote_copy = quote.copy()
quote_copy.TradingDay = quote_copy.TradingDay.apply(lambda x: str(x)[:7])
quote_copy = quote_copy.groupby(['SecuCode', 'TradingDay']).apply(
    lambda x: (x.Close.values[-1] - x.Close.values[0]) / x.Close.values[0])
quote_oc = pd.DataFrame(quote_copy)
quote_oc.columns = ['Ratio']


records_temp = records_month.reset_index()
codes = set(records.SecuCode.unique()) - set(quote.SecuCode.unique())
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
industry_copy = industry_quote.copy()
industry_copy.TradingDay = industry_quote.TradingDay.apply(lambda x: str(x)[:7])
industry_copy = industry_copy.set_index('TradingDay')
ind = industry_copy.index
industry_copy.columns.names = ['Industry']
industry_copy = industry_copy.melt()
industry_copy.index = ind.tolist() * 29
industry_copy = industry_copy.reset_index()
industry_copy.columns = ['TradingDay', 'Industry', 'Close']

industry_month_ratio = industry_copy.groupby(['Industry', 'TradingDay']).apply(
        lambda x: (x.Close.values[-1] - x.Close.values[0]) / x.Close.values[0])
industry_month_ratio = pd.DataFrame(industry_month_ratio)
industry_month_ratio.columns = ['Ratio']


# combine industry data with quote data
industry_month_ratio1 = industry_month_ratio.reset_index()
industry_month_ratio1.columns = ['SecuCode', 'TradingDay', 'Ratio']
quote_temp = quote_oc.reset_index()
quote_temp = pd.concat([quote_temp, industry_month_ratio1])


market = records_temp.groupby(['PortCode']).apply(lambda x: quote_temp[quote_temp.SecuCode.isin(
                x.SecuCode.unique())]).reset_index()
market_month_ratio = market.groupby(['PortCode', 'TradingDay']).apply(lambda x: x.Ratio.mean())
market_month_ratio = pd.DataFrame(market_month_ratio).reset_index()


# merge market_month_ratio to nav datset in order to compare with nav_month_ratio
nav_temp = nav_copy.groupby(['PortCode', 'NavDate']).apply(lambda x: x.iloc[0])
nav_temp = nav_temp.drop(['NavDate', 'PortCode'], axis=1).reset_index()
market_month_ratio = nav_temp.merge(market_month_ratio, how='left', left_on=['PortCode', 'NavDate'], 
                                                  right_on=['PortCode', 'TradingDay'])
market_month_ratio.drop(['TradingDay', 'ID', 'Nav'], axis=1, inplace=True)
market_month_ratio.columns = ['PortCode', 'NavDate', 'Ratio']


nav_month_ratio_reset = nav_month_ratio.reset_index()
full = market_month_ratio.merge(nav_month_ratio_reset, 
                                 left_on=['PortCode', 'NavDate'], right_on=['PortCode', 'NavDate'])
full['excess'] = full.Ratio_y - full.Ratio_x
result = pd.DataFrame(full.groupby(['PortCode']).apply(lambda x: x.excess.mean() / x.excess.std()))
result.columns = ['IR']


IR_rank = result[~pd.isnull(result.IR)].sort_values(['IR'])

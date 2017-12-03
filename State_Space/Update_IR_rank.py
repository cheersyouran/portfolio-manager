
import numpy as np
import pandas as pd
import base
from datetime import datetime

industry = base.load_industry_csv()


def year_week(x):
    year = str(pd.Timestamp(x).isocalendar()[0])
    week = str(pd.Timestamp(x).isocalendar()[1])
    if len(week)==1:
        week = '0' + week
    return year + '-' + week


def generate_port_stock_mapping(df_records, df_quote):
    codes = set(df_records.SecuCode.unique()) - set(df_quote.SecuCode.unique())
    codes = list(codes.intersection(set(industry.SecuCode.unique())))
    temp = industry[industry.SecuCode.isin(codes)].set_index(['SecuCode'])
    dicts = temp.to_dict()['FirstIndustryName']

    port_stock_mapping = \
        pd.DataFrame(df_records.groupby(['PortCode']).apply(lambda x: pd.Series(x.SecuCode.unique())))
    port_stock_mapping.columns = ['Secu_Code']
    port_stock_mapping['SecuCode'] = port_stock_mapping.Secu_Code.map(dicts)
    port_stock_mapping.SecuCode.fillna(port_stock_mapping.Secu_Code, inplace=True)
    port_stock_mapping.drop(['Secu_Code'], axis=1, inplace=True) 
    port_stock_mapping = pd.DataFrame(port_stock_mapping.reset_index().groupby(['PortCode']).apply(lambda x: 
                                                                pd.Series(x.SecuCode.unique()))).reset_index()
    return port_stock_mapping, dicts


def compute_std(N_after, s, mean, mean_after, new):
    nom = (N_after-1)*s*s + (new-mean)*(new-mean_after)
    denom = N_after
    return np.sqrt(nom/denom)



# Demo:
#     IR_rank = Update_IR_rank(date, df_records, df_ind_quote, df_nav, df_quote, IR_rank, save_path='./data')
#     For a new coming day, given new records, quote, industryquote, nav and previous IR_rank data, update the IR_rank data and save it. 

   

def Update_IR_rank(date, df_records, df_ind_quote, df_nav, df_quote, IR_rank, save_path=None):
    # Decide whether this date is the end of the week. If not return the original IR rank.
    if datetime.weekday(np.datetime64(date).item()) != 4:
        return IR_rank
    date = np.datetime64(date)
    dates = [str(date-np.timedelta64(i)) for i in range(0, 5)]
    
    # Get sub data for the specific week.
    # Not in records does not mean the portfolio does not exist as long as it's in nav dataset.
    port_stock_mapping, dicts = generate_port_stock_mapping(df_records, df_quote)
    
    df_nav.NavDate = df_nav.NavDate.apply(lambda x: str(x)[:10])
    new_nav = df_nav.loc[df_nav.NavDate.isin(dates)]
    new_nav.NavDate = new_nav.NavDate.apply(year_week)
    df_records.Updated = df_records.Updated.apply(lambda x: str(x)[:10])
    new_records = df_records.loc[df_records.Updated.isin(dates)]
    new_records.Updated = new_records.Updated.apply(year_week)
    df_quote.TradingDay = df_quote.TradingDay.apply(lambda x: str(x)[:10])
    new_quote = df_quote.loc[df_quote.TradingDay.isin(dates)]
    new_quote.TradingDay = new_quote.TradingDay.apply(year_week)
    df_ind_quote.TradingDay = df_ind_quote.TradingDay.apply(lambda x: str(x)[:10])
    new_ind_quote = df_ind_quote.loc[df_ind_quote.TradingDay.isin(dates)]
    new_ind_quote.TradingDay = new_ind_quote.TradingDay.apply(year_week)
    
    new_records['Secu_Code'] = new_records.SecuCode.map(dicts)
    new_records.Secu_Code.fillna(new_records.SecuCode, inplace=True)
    new_records.drop(['SecuCode', 'PrevWeight', 'TargetWeight', 'StockName'], axis=1, inplace=True)
    
    new_quote = new_quote.groupby(['SecuCode', 'TradingDay']).apply(
        lambda x: (x.Close.values[-1] - x.Close.values[0]) / x.Close.values[0])
    new_quote = pd.DataFrame(new_quote)
    new_quote.columns = ['Ratio']
    
    new_ind_quote = new_ind_quote.set_index('TradingDay')
    ind = new_ind_quote.index
    new_ind_quote.columns.names = ['Industry']
    new_ind_quote = new_ind_quote.melt()
    new_ind_quote.index = ind.tolist() * 29
    new_ind_quote = new_ind_quote.reset_index()
    new_ind_quote.columns = ['TradingDay', 'Industry', 'Close']
    
    new_industry_week_ratio = new_ind_quote.groupby(['Industry', 'TradingDay']).apply(
        lambda x: (x.Close.values[-1] - x.Close.values[0]) / x.Close.values[0])
    new_industry_week_ratio = pd.DataFrame(new_industry_week_ratio)
    new_industry_week_ratio.columns = ['Ratio']
    
    new_industry_week_ratio1 = new_industry_week_ratio.reset_index()
    new_industry_week_ratio1.columns = ['SecuCode', 'TradingDay', 'Ratio']
    new_quote_temp = new_quote.reset_index()
    new_quote_temp = pd.concat([new_quote_temp, new_industry_week_ratio1])
    
    # Consider all existing portcodes and their stocks
    portcodes_exist = pd.Series(new_nav.PortCode.unique())
    portcodes_exist = portcodes_exist[~portcodes_exist.isin(new_records.PortCode)]
    new_port_stock_mapping = port_stock_mapping[port_stock_mapping.PortCode.isin(portcodes_exist)]
    new_port_stock_mapping.columns = ['PortCode', 'Updated', 'Secu_Code']
    new_records = pd.concat([new_records, new_port_stock_mapping])
    new_market = new_records.groupby(['PortCode']).apply(lambda x: new_quote_temp[new_quote_temp.SecuCode.isin(
                x.Secu_Code.unique().astype('object'))]).reset_index()
    new_market_week_ratio = pd.DataFrame(new_market.groupby(['PortCode', 'TradingDay']).apply(
                                                            lambda x: x.Ratio.mean())).reset_index()
    
    new_nav_week_ratio = pd.DataFrame(new_nav.groupby(['PortCode', 'NavDate']).apply(
        lambda x: (x.Nav.values[-1] - x.Nav.values[0])/x.Nav.values[0])).reset_index()
    new_nav_week_ratio.columns = ['PortCode', 'NavDate', 'Ratio']
    
    new_nav_temp = new_nav.groupby(['PortCode', 'NavDate']).apply(lambda x: x.iloc[0])
    new_nav_temp = new_nav_temp.drop(['NavDate', 'PortCode'], axis=1).reset_index()
    new_market_week_ratio = new_nav_temp.merge(new_market_week_ratio, how='left', left_on=['PortCode', 'NavDate'], 
                                                      right_on=['PortCode', 'TradingDay'])
    new_market_week_ratio.drop(['TradingDay', 'ID', 'Nav'], axis=1, inplace=True)
    new_market_week_ratio.columns = ['PortCode', 'NavDate', 'Ratio']
    new_full = new_market_week_ratio.merge(new_nav_week_ratio, 
                                 left_on=['PortCode', 'NavDate'], right_on=['PortCode', 'NavDate'])
    new_full['excess'] = new_full.Ratio_y - new_full.Ratio_x
    new_full.dropna(inplace=True)
    
    # update current IR rank
    new_IR_rank = IR_rank.reset_index()

    # If there's a new portfolio appears.
    # In its first week, set IR to 0.
    new_portcodes = np.array(list(set(new_full.PortCode)-set(new_IR_rank.PortCode)))
    if len(new_portcodes) > 0:
        new_row = pd.DataFrame(np.zeros((len(new_portcodes), 5)))
        new_row.iloc[:, 0] = new_portcodes
        new_row.columns = IR_rank.reset_index().columns.tolist()
        new_IR_rank.append(new_row)


    new_IR_rank['total'] = new_IR_rank['num_week'] * new_IR_rank['mean']
    new_IR_rank = pd.merge(new_IR_rank, new_full, how='left', left_on='PortCode', right_on='PortCode')
    new_IR_rank = new_IR_rank.drop(['NavDate', 'Ratio_x', 'Ratio_y'], axis=1)
    
    ind = np.where(new_IR_rank.PortCode.isin(new_full.PortCode))[0]
    new_IR_rank['num_week'][ind] += 1
    new_IR_rank['mean_after'] = (new_IR_rank.total + new_IR_rank.excess)/(new_IR_rank.num_week)
    new_IR_rank['std_after'] = compute_std(new_IR_rank['num_week'], new_IR_rank['std'], 
                                       new_IR_rank['mean'], new_IR_rank['mean_after'], new_IR_rank['excess'])

    new_IR_rank['IR_after'] = new_IR_rank['mean_after'] / (new_IR_rank['std_after'] + 1e-20)
    new_IR_rank.loc[np.abs(new_IR_rank.IR_after) > 1000, 'IR_after'] = 0
    new_IR_rank['mean_after'] = new_IR_rank['mean_after'].fillna(new_IR_rank['mean'])
    new_IR_rank['std_after'] = new_IR_rank['std_after'].fillna(new_IR_rank['std'])
    new_IR_rank['IR_after'] = new_IR_rank['IR_after'].fillna(new_IR_rank['IR'])
    new_IR_rank.drop(['IR', 'mean', 'std', 'total', 'excess'], axis=1, inplace=True)
    new_IR_rank.columns = ['PortCode', 'num_week', 'mean', 'std', 'IR']
    new_IR_rank = new_IR_rank[~pd.isnull(new_IR_rank.IR)].sort_values(['IR'])
    
    if save_path is None:
        return new_IR_rank
    else:
        new_IR_rank.to_csv(save_path + '/IR_rank_week.csv')
        print('Saved updated IR_rank_week to', save_path)
        return new_IR_rank
    
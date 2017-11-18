import base
import numpy as np
import pandas as pd
import talib as tb
import stockstats as st
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score


industry = base.load_industry_csv()


def search_industries(port, df_records):
    secucodes = df_records[df_records.PortCode == port].SecuCode.unique()
    dicts = industry.set_index(['SecuCode']).to_dict()['FirstIndustryName']
    industries = pd.Series(secucodes).map(dicts).unique()
    return pd.Series(industries).dropna().values


# Demo:
#   search_port('银行', df_records, df_nav, IR_rank, ['ZH000199', 'ZH010630'], output=4)
#   given records and IR rank data, from given portfolio list return 4 good portfolios (in the first 300 of IR rank) which trade on '银行' and still exist.

def search_port(indust, df_records, df_nav, IR_rank, port_list=None, output=5):
    mapping = industry.set_index('SecuCode').to_dict()['FirstIndustryName']
    records_copy = df_records.copy()
    records_copy.SecuCode = records_copy.SecuCode.map(mapping)
    records_copy.drop('StockName', 1, inplace=True)
    records_copy.columns = ['PortCode', 'Updated', 'Industry', 'PrevWeight', 'TargetWeight']
    records_copy.Updated = records_copy.Updated.apply(lambda x: str(x)[:10])
    records_copy.dropna(inplace=True)
    portcodes = records_copy[records_copy.Industry == indust].PortCode.unique()
    if port_list is not None:
        portcodes = portcodes[pd.Series(portcodes).isin(port_list)]
    last_day = np.sort(df_nav.NavDate.unique())[-1]
    ind2 = pd.Series(nav[nav.NavDate == last_day].PortCode.unique())
    portcodes = ind2[ind2.isin(portcodes)]
    ind = IR_rank.iloc[-300:].index
    portcodes = ind[ind.isin(portcodes)]
    return np.array(portcodes)[-output:][::-1]


# For each portfolio, count how many times(days) they traded on one industry and then normalize.
# Values between 0 and 1. 
def generate_freq_table(IR_rank, df_records):
    ind = IR_rank.iloc[-300:].index.values
    records_sub = df_records[df_records.PortCode.isin(ind)].copy()
    dicts = industry.set_index(['SecuCode']).to_dict()['FirstIndustryName']
    records_sub.SecuCode = records_sub.SecuCode.map(dicts)
    records_sub.drop('StockName', 1, inplace=True)
    records_sub.columns = ['PortCode', 'Updated', 'Industry', 'PrevWeight', 'TargetWeight']
    records_sub.Updated = records_sub.Updated.apply(lambda x: str(x)[:10])
    records_sub.dropna(inplace=True)
    records_sub = pd.DataFrame(records_sub.groupby(['PortCode', 'Updated']).apply(lambda x: 
                                                            pd.Series(x.Industry.unique()))).reset_index()
    records_sub.drop(['level_2'], axis=1, inplace=True)
    records_sub.columns = ['PortCode', 'Updated', 'Industry']
    port_num = pd.DataFrame(records_sub.groupby(['PortCode', 'Industry']).apply(lambda x: 
                                                                                len(x))).reset_index()
    port_num.set_index(['PortCode', 'Industry'], inplace=True)
    port_num = port_num.unstack().fillna(0).applymap(lambda x: int(x))
    port_num.columns = port_num.columns.levels[1]
    port_num = port_num.div(port_num.sum(1), axis=0)
    
    return port_num


# For each portfolio, see whether they trade on one industry or not.
# Values either 0 or 1.
def generate_dummy_table(IR_rank, df_records):
    ind = IR_rank.iloc[-300:].index.values
    records_sub = df_records[df_records.PortCode.isin(ind)].copy()
    dicts = industry.set_index(['SecuCode']).to_dict()['FirstIndustryName']
    records_sub.SecuCode = records_sub.SecuCode.map(dicts)
    records_sub.drop('StockName', 1, inplace=True)
    records_sub.columns = ['PortCode', 'Updated', 'Industry', 'PrevWeight', 'TargetWeight']
    records_sub.Updated = records_sub.Updated.apply(lambda x: str(x)[:10])
    records_sub.dropna(inplace=True)
    records_sub.index = records_sub.PortCode
    records_sub.drop(['PortCode', 'Updated', 'PrevWeight', 'TargetWeight'], 1, inplace=True)
    Port_Region = pd.get_dummies(records_sub, prefix='', prefix_sep='')
    Port_Region = Port_Region.groupby('PortCode').agg(lambda x: np.sign(x.sum()))
    return Port_Region



# find similar portfolios using frequency table.
# Demo:
#   find_similar_ports_byFreq('ZH000199', df_records, IR_rank, ['ZH000283', 'ZH877294', 'ZH395487', 'ZH015838'], output=4)
# given records data and IR rank data, from the portcodes list, return 4 portcodes that are most similar to ZH000199

def find_similar_ports_byFreq(port, df_records, IR_rank, port_list=None, thresh=0.5, output=5):
    port_num = generate_freq_table(IR_rank, df_records)
    if port_list is not None:
        ind = np.array(port_list)
        ind = ind[pd.Series(ind).isin(port_num.index)]
        port_num = port_num.loc[ind]
    dicts = industry.set_index(['SecuCode']).to_dict()['FirstIndustryName']
    records_copy = df_records[df_records.PortCode==port].copy()
    records_copy.SecuCode = records_copy.SecuCode.map(dicts)
    records_copy.columns = ['PortCode', 'Updated', 'Industry', 'StockName','PrevWeight', 'TargetWeight']
    records_copy = pd.DataFrame(records_copy.groupby(['PortCode', 'Updated']).apply(lambda x: 
                                                                pd.Series(x.Industry.unique()))).reset_index()
    records_copy.columns = np.insert(records_copy.columns.values[:-1], len(records_copy.columns)-1, 'Industry')
    port_num_test = pd.DataFrame(records_copy.groupby(['PortCode', 'Industry']).apply(lambda x: 
                                                                                    len(x))).reset_index()
    col = port_num.columns
    test = np.array([0]*29)
    port_num_test.columns = ['PortCode', 'Industry', 'Freq']
    for i in range(len(port_num_test.Industry.values)):
        val = port_num_test.Industry.values[i]
        test[np.where(col==val)[0]] = port_num_test.iloc[i, :].Freq
    scores, ports_li = [], []
    test = test/np.sum(test)
    for i in range(len(port_num)):
        scores.append(1-cosine(port_num.iloc[i, :].values.reshape((-1, 1)), test.reshape((-1, 1))))
        ports_li.append(port_num.index[i])
    ports_scores = pd.DataFrame([ports_li, scores]).T
    ports_scores.columns = ['PortCode', 'Score']
    ports_scores.sort_values(['Score'], inplace=True)
    ports_selected = ports_scores[ports_scores.Score>=thresh].loc[ports_scores.Score<0.98]
    return ports_selected.PortCode.values[::-1][:output]



# find similar portfolios using dummy table.
# Demo:
#   find_similar_ports_byRegion('ZH000199', output=4)
# return 4 portcodes that are most similar to ZH000199

def find_similar_ports_byRegion(port, df_records, IR_rank, thresh=0.5, output=5):
    Port_Region = generate_dummy_table(IR_rank, df_records)
    dicts = industry.set_index(['SecuCode']).to_dict()['FirstIndustryName']
    records_copy = records[records.PortCode==port].copy()
    records_copy.StockName = records_copy.SecuCode.map(dicts)
    industries = pd.Series(records_copy.StockName.unique()).dropna().values
    test = Port_Region.columns.isin(industries).astype('int')
    scores, ports_li = [], []
    for i in range(len(Port_Region)):
        scores.append(f1_score(Port_Region.iloc[i, :].values, test))
        ports_li.append(Port_Region.index[i])
    ports_scores = pd.DataFrame([ports_li, scores]).T
    ports_scores.columns = ['PortCode', 'Score']
    ports_scores.sort_values(['Score'], inplace=True)
    ports_selected = ports_scores[ports_scores.Score>=thresh].loc[ports_scores.Score<1]
    return ports_selected.PortCode.values[::-1][:output]    



def func(x):
    x = 1 if x>=0 else 0
    return x

def func2(x):
    if x<=20:
        x = 1
    elif x>=80:
        x = -1
    else:
        x = 0
    return x

def func3(x):
    if x<=0:
        x = 1
    elif x>=1:
        x = -1
    else:
        x = 0
    return x

def func4(x):
    if x<=0.1:
        x = 1
    else:
        x = 0
    return x
  



# Rolling Average, MACD, BOLL, KDJ
# Demo:
#   generate_states('银行', df_industry_quote, ['MACD', 'KDJ', [5, 10], [10, 15], 'BOLL'])
#   given industry quote data, generate MACD, KDJ, 5-10-average, 10-15-average and BOLL of '银行' industry.
def generate_states(industry_name, df_industry_quote, params):
    assert params.__class__ is list, 'Params must be in list form.'
    tradingday = df_industry_quote.TradingDay.values
    
    def ave(name, day1, day2):
        roll_day1 = df_industry_quote.loc[:, name].rolling(day1).mean()
        roll_day2 = df_industry_quote.loc[:, name].rolling(day2).mean()
        roll_diff = roll_day2 - roll_day1
        roll = pd.DataFrame(roll_diff)
        roll.columns = [name + '_' + str(day1) + '_' + str(day2)]
        roll.index = tradingday
        roll = pd.DataFrame(roll.iloc[:, 0].apply(func))
        return roll[day2-1:]
    
    def macd(name):
        vals = df_industry_quote.loc[:, name].values.astype('float')
        dif = tb.MACD(vals)[0]
        dem = tb.MACD(vals)[1]
        macd = pd.DataFrame({'DIF_diff':dif, 'DIF_DEM':dem}, index=tradingday)
        macd.DIF_DEM = macd.DIF_diff - macd.DIF_DEM
        macd.DIF_diff = macd.DIF_diff.diff()
        macd.DIF_DEM = macd.DIF_DEM.apply(func)
        macd.DIF_diff = macd.DIF_diff.apply(func)
        return macd[34:]
    
    def boll(name):
        vals = pd.DataFrame(df_industry_quote.loc[:, name])
        vals.columns = ['close']
        stdf = st.StockDataFrame.retype(vals)
        boll = stdf['boll']
        boll_up = stdf['boll_ub']
        boll_down = stdf['boll_lb']
        b = (vals.close-boll_down).div(boll_up-boll_down)
        bdwidth = (boll_up-boll_down).div(boll)
        BOLL = pd.DataFrame({'percent_b':b.values, 'bdwidth':bdwidth.values}, index=tradingday)
        BOLL.percent_b = BOLL.percent_b.apply(func3)
        BOLL.bdwidth = BOLL.bdwidth.apply(func4)
        return BOLL[1:]
    
    def kdj(name):
        vals = pd.DataFrame(df_industry_quote.loc[:, name]).applymap(lambda x: float(x))
        vals.columns = ['close']
        vals['low'] = vals.close.rolling(5).min()
        vals['high'] = vals.close.rolling(5).max()
        for i in range(5):
            vals.low.loc[i] = np.min(vals.close.values[:i+1])
            vals.high.loc[i] = np.max(vals.close.values[:i+1])
        k, d = tb.STOCH(vals.high.values, vals.low.values, vals.close.values.astype('float'))
        KDJ = pd.DataFrame({'KDJ_K':k, 'KDJ_D':d}, index=tradingday)
        KDJ.KDJ_K = KDJ.KDJ_K.apply(func2)
        KDJ.KDJ_D = KDJ.KDJ_D.apply(func2)
        return KDJ[8:]
    
    states = pd.DataFrame([0]*len(tradingday), index=tradingday)
    name = industry_name
    for feature in params:
        if feature.__class__ is list:
            states = states.merge(ave(industry_name, feature[0], feature[1]), 
                                  left_index=True, right_index=True)
            name += '_' + str(feature[0]) + '_' + str(feature[1])
        elif feature == 'MACD':
            states = states.merge(macd(industry_name), left_index=True, right_index=True)
            name += '_MACD'
        elif feature == 'BOLL':
            states = states.merge(boll(industry_name), left_index=True, right_index=True)
            name += '_BOLL'
        elif feature == 'KDJ':
            states = states.merge(kdj(industry_name), left_index=True, right_index=True)
            name += '_KDJ'
    output = states.iloc[:, 1:]
    output.index.names = ['TradingDay']
    output.to_csv(name+'.csv')
    return output



# Demo:
#   periods, count = get_active_periods('ZH000199', df_records)

def get_active_periods(port, df_records):
    periods = df_records[df_records.PortCode == port].Updated.apply(lambda x: str(x)[:10])
    periods = pd.DataFrame(periods.unique())
    periods.columns = [port + '_Periods']
    count = len(periods)
    return periods, count

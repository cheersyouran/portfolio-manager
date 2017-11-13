import base
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


IR_rank = base.load_irweek_csv()
quote = base.load_quote_csv()
industry = base.load_industry_csv()
records = base.load_records_csv()
industry_quote = base.load_industryquote_csv()


def search_industies(port):
    secucodes = records[records.PortCode == port].SecuCode.unique()
    dicts = industry.set_index(['SecuCode']).to_dict()['FirstIndustryName']
    industries = pd.Series(secucodes).map(dicts).unique()
    return pd.Series(industries).dropna().values


def generate_table():
    ind = IR_rank.iloc[-300:].index.values
    records_sub = records[records.PortCode.isin(ind)].copy()
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


def find_similar_ports(port, thresh=0.5):
    Port_Region = generate_table()
    dicts = industry.set_index(['SecuCode']).to_dict()['FirstIndustryName']
    records_copy = records[records.PortCode==port].copy()
    records_copy.StockName = records_copy.SecuCode.map(dicts)
    industries = pd.Series(records_copy.StockName.unique()).dropna().values
    test = Port_Region.columns.isin(industries).astype('int')
    scores = []
    for i in range(len(Port_Region)):
        scores.append(f1_score(Port_Region.iloc[i, :].values, test))
    scores = np.sort(scores)
    ports_li = Port_Region.index[np.where(scores>=thresh)[0]].values
    ports_li = [Port for Port in ports_li if Port != port]
    return ports_li[-5:]


def generate_states(industry_name, feature):
    
    def ave(name, day1, day2):
        roll_day1 = industry_quote.loc[:, name].rolling(day1).mean()
        roll_day2 = industry_quote.loc[:, name].rolling(day2).mean()
        roll_diff = roll_day2 - roll_day1
        tradingday = industry_quote.TradingDay.values
        roll = pd.DataFrame(roll_diff)
        roll.index = tradingday
        return roll[day2-1:]
    
    if feature.__class__ is list:
        return ave(industry_name, feature[0], feature[1])
    else:
        return None

if __name__ == '__main__':

    ind = search_industies('ZH010630')
    ps = find_similar_ports('ZH010630')
    s = generate_states('银行', [2, 5])

    print(ind)
    print(ps)
    print(s)
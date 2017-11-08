
import pandas as pd
import numpy as np


quote = pd.read_csv('quote.csv', sep='\t', parse_dates=['TradingDay'])
industry_quote = pd.read_csv('/Users/wangchengming/Documents/5001Project/Snowball/EvaluationDemo/industry_quote.csv', parse_dates=['tradingday'])
# Transform the original xml file into csv format. Then read in.
industry_quote_full = pd.read_csv('industry_quote.csv', encoding='gbk')


ind = industry_quote_full.loc[:0, :].values[:, 1:].tolist()[0]
ind = [i[:-4] for i in ind]
industry_quote.columns = ['TradingDay'] + ind


# 10-day rolling mean minus 5-day rolling mean
rolling_mean_iron = industry_quote.loc[:, '钢铁'].rolling(10).mean()[9:] - \
                    industry_quote.loc[:, '钢铁'].rolling(5).mean()[9:]
rolling_mean_bank = industry_quote.loc[:, '银行'].rolling(10).mean()[9:] - \
                    industry_quote.loc[:, '银行'].rolling(5).mean()[9:]
rolling_mean_computer = industry_quote.loc[:, '计算机'].rolling(10).mean()[9:] - \
                        industry_quote.loc[:, '计算机'].rolling(5).mean()[9:]
rolling_mean_market = quote.groupby('TradingDay').Close.mean().rolling(10).mean()[9:-51] - \
                      quote.groupby('TradingDay').Close.mean().rolling(5).mean()[9:-51]
TradingDay = industry_quote.TradingDay[9:]


dicts = {'Bank': rolling_mean_bank, 'TradingDay': TradingDay,
         'Iron': rolling_mean_iron, 'Computer': rolling_mean_computer,
         'Market': rolling_mean_market.values}
rolling_data = pd.DataFrame(data=dicts, columns=['TradingDay', 'Bank', 'Iron', 'Computer', 'Market'])
rolling_data.index = np.arange(len(TradingDay))

rolling_data.to_csv('rolling_data1.csv', index=False)
import pandas as pd
import numpy as np

# 10-day rolling mean minus 5-day rolling mean
ROLL_DAY1 = 10
ROLL_DAY2 = 5
IND = np.max([ROLL_DAY1, ROLL_DAY2]) - 1


quote = pd.read_csv('/Users/wangchengming/Documents/5001Project/Snowball/RL/quote.csv', sep='\t', parse_dates=['TradingDay'])
industry_quote = pd.read_csv('/Users/wangchengming/Documents/5001Project/Snowball/EvaluationDemo/industry_quote.csv', parse_dates=['tradingday'])
# Transform the original xml file into csv format. Then read in.
industry_quote_full = pd.read_csv('/Users/wangchengming/Documents/5001Project/Snowball/RL/industry_quote.csv', encoding='gbk')


ind = industry_quote_full.loc[:0, :].values[:, 1:].tolist()[0]
ind = [i[:-4] for i in ind]
industry_quote.columns = ['TradingDay'] + ind


rolling_mean_1 = industry_quote.loc[:, '银行'].rolling(ROLL_DAY1).mean()[IND:] - \
                    industry_quote.loc[:, '银行'].rolling(ROLL_DAY2).mean()[IND:]
rolling_mean_2 = industry_quote.loc[:, '家电'].rolling(ROLL_DAY1).mean()[IND:] - \
                    industry_quote.loc[:, '家电'].rolling(ROLL_DAY2).mean()[IND:]
# rolling_mean_market = quote.groupby('TradingDay').Close.mean().rolling(10).mean()[9:-51] - \
                      # quote.groupby('TradingDay').Close.mean().rolling(5).mean()[9:-51]
TradingDay = industry_quote.TradingDay[IND:]


dicts = {'Bank': rolling_mean_1, 'TradingDay': TradingDay,
         'Appliance': rolling_mean_2}
rolling_data = pd.DataFrame(data=dicts, columns=['TradingDay', 'Bank', 'Appliance'])
rolling_data.index = np.arange(len(TradingDay))

rolling_data.to_csv('rolling_data1.csv', index=False)
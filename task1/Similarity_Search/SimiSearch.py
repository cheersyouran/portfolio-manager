
import pandas as pd
import numpy as np
import base
from task1.Similarity_Search.utils import *
from bitarray import bitarray
from task1.Similarity_Search.CONFIG import *

test_date = base.TASK1_DATE
quote.TradingDay = quote.TradingDay.map(lambda x: str(x)[:10])


mapping = industry.set_index(['SecuCode']).to_dict()['FirstIndustryName']
ind_sec = pd.DataFrame(industry.groupby(['FirstIndustryName']).apply(lambda x:
                                                    x.SecuCode[x.SecuCode.isin(quote.SecuCode)].unique()))
ind_sec.columns = ['SecuCodes']
mapping_reverse = ind_sec.to_dict()['SecuCodes']


# Some codes have no industry information. Just drop them.
code_ind = quote[~quote.SecuCode.isin(industry.SecuCode)].SecuCode.unique()
quote = quote[~quote.SecuCode.isin(code_ind)]
# quote_sub contains pre-saved historical matrices and quote_test contains future incoming matrices.
test_start_ind = int(np.where(quote.TradingDay.unique() == test_date)[0])
date_sub_range = quote.TradingDay.unique()[:test_start_ind]
quote_sub = quote[quote.TradingDay.isin(date_sub_range)]
date_test_range = quote.TradingDay.unique()[test_start_ind:]
quote_test = quote[quote.TradingDay.isin(date_test_range)]
test_end_ind = len(quote.TradingDay.unique())

if not LIMITED:
    NUM_DAY = test_end_ind - test_start_ind

print('\n################ Number of Days to test:', NUM_DAY, '################')

print('\n################ Start Constructing Historical Database... ###############\n')

# Save historical matrices as dicts and bitmap arrays.
# Dictionary keys are industry names and its values are sub-dictionaries with security codes as keys.
i = 0
signs_dict = {}
for ind in industry.FirstIndustryName.unique():
    time1 = time.time()
    signs_dict[ind] = {}
    for secu in mapping_reverse[ind]:
        signs_mat = signs_window(quote_sub, secu, WINDOW_SIZE)
        signs_bit = bitarray(signs_mat.flatten().tolist())
        signs_dict[ind][secu] = signs_bit
        i += 1
        print('SecuCode:', secu, '+++++', 'Num:', i)
    time2 = time.time()
    print('----------------- Industry:', ind, ' ---- ', 'Time Spent:', round(time2-time1, 2))

print('\n################ Historical Database has been contructed. #################')

print('\n################ Start testing on test set...             ###################')

# Main test phase. By iteratively updating and searching, get the top secucodes for each day and assign weights to them.
top = pd.DataFrame([0, 0, 0]).T
top.columns = ['tradingday', 'stockcode', 'stockweight']
for i in range(NUM_DAY):
    date_  = quote_test.TradingDay.iloc[i]
    date = quote_test.TradingDay.iloc[i+1]
    update_all_signs_bit(date_, quote_test, mapping, signs_dict, WINDOW_SIZE)
    secu_codes = quote_test[quote_test.TradingDay == date].SecuCode.unique()
    secu_probs = {}
    for code in secu_codes[:NUM_CODES]:
        indust = mapping[code]
        signs_bit = signs_dict[indust][code]
        vec = np.array(list(signs_bit[-WINDOW_SIZE:].to01())).astype('int')
        
        secu_distribution = []
        for secu in find_similar_secu(code, mapping, mapping_reverse):
            signs_bit = signs_dict[indust][secu]
            signs_mat = np.array(list(signs_bit.to01())).astype('int').reshape((-1, WINDOW_SIZE))
            secu_distribution.extend(get_distribution(signs_mat, vec, front=0.2))

        secu_distribution = np.array(secu_distribution[:100])
        probs = np.ones(100) / 100
        secu_prob = float(secu_distribution.reshape((1, -1)).dot(probs.reshape((-1, 1))))
        secu_probs[code] = secu_prob
        print('Currently Testing...  ', 'Date:', date_, '----', 'SecuCode:', code, '----')
    top_codes = generate_weights(secu_probs, date, top=TOP)
    top = pd.concat([top, top_codes])

top = top[1:]
top.tradingday = top.tradingday.apply(lambda x: str(x)[:10])
top.sort_values(['tradingday'], inplace=True)

path = '/Users/wangchengming/Documents/5001Project/Snowball/EvaluationDemo/Task1_output.csv'
top.to_csv(path, index=None)

print('\n#################### Stock Weight file is saved in', path, '####################')
# base_weights = generate_baseline_weights_2(NUM_DAY, 10)
# base_weights.to_csv('/Users/wangchengming/Documents/5001Project/Snowball/EvaluationDemo/task1_baseline2.csv', 
                     # index=None)
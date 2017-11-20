
import pandas as pd
import numpy as np
import base
from utils import *
from bitarray import bitarray


# window_size should be the same as those in the historical matrices.
NUM_DAY = 20
NUM_CODES = 200
TOP = 6
WINDOW_SIZE = 10


quote = base.load_quote_csv()
industry = base.load_industry_csv()
quote.dropna(inplace=True)


# Some codes have no industry information. Just drop them.
code_ind = quote[~quote.SecuCode.isin(industry.SecuCode)].SecuCode.unique()
quote = quote[~quote.SecuCode.isin(code_ind)]
# quote_sub contains pre-saved historical matrices and quote_test contains future incoming matrices.
date_sub = quote.TradingDay.unique()[:-NUM_DAY-2]
quote_sub = quote[quote.TradingDay.isin(date_sub)]
date_test = quote.TradingDay.unique()[-NUM_DAY-2:]
quote_test = quote[quote.TradingDay.isin(date_test)]


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
    print('Industry:', ind, ' ---- ', 'Time Spent:', round(time2-time1, 2))


# Main test phase. By iteratively updating and searching, get the top secucodes for each day and assign weights to them.
top = pd.DataFrame([0, 0, 0]).T
top.columns = ['tradingday', 'stockcode', 'stockweight']
for i in range(NUM_DAY):
    date_  = quote_test.TradingDay.iloc[i]
    date = quote_test.TradingDay.iloc[i+1]
    update_all_signs_bit(date_, WINDOW_SIZE)
    secu_codes = quote_test[quote_test.TradingDay == date].SecuCode.unique()
    secu_probs = {}
    for code in secu_codes[:NUM_CODES]:
        indust = mapping[code]
        signs_bit = signs_dict[indust][code]
        vec = np.array(list(signs_bit[-WINDOW_SIZE:].to01())).astype('int')
        
        secu_distribution = []
        for secu in find_similar_secu(code):
            signs_bit = signs_dict[indust][secu]
            signs_mat = np.array(list(signs_bit.to01())).astype('int').reshape((-1, WINDOW_SIZE))
            secu_distribution.extend(get_distribution(signs_mat, vec, front=0.2))

        secu_distribution = np.array(secu_distribution[:100])
        probs = np.ones(100) / 100
        secu_prob = float(secu_distribution.reshape((1, -1)).dot(probs.reshape((-1, 1))))
        secu_probs[code] = secu_prob
        print('Date:', date_, '----', 'SecuCode:', code, '----', 'Prob:', secu_prob)
    top_codes = generate_weights(secu_probs, date, top=TOP)
    top = pd.concat([top, top_codes])

top = top[1:]
top.tradingday = top.tradingday.apply(lambda x: str(x)[:10])


# top.to_csv('/Users/wangchengming/Documents/5001Project/Snowball/EvaluationDemo/task1_my_input3.csv', index=None)

base_weights = generate_baseline_weights_2(NUM_DAY, 10)
# base_weights.to_csv('/Users/wangchengming/Documents/5001Project/Snowball/EvaluationDemo/task1_baseline2.csv', 
                     # index=None)
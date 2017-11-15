
import pandas as pd
import numpy as np
import datetime
import time
from bitarray import bitarray
import base

quote = base.load_quote_csv()
industry = base.load_industry_csv()
quote.dropna(inplace=True)


def func(x):
    if x < 0:
        return 0
    elif x >= 0:
        return 1

def signs_window(quote, code, size=10):
    vals = quote[quote.SecuCode == code].Close
    signs = pd.DataFrame(vals.diff()[1:].apply(func))
    names = ['signs_1']
    df = signs
    for i in range(1, size):
        df = df.merge(signs.shift(-i), left_index=True, right_index=True)
        names.append('signs_' + str(i+1))
    df.columns = names
    return np.array(df.dropna().applymap(lambda x: int(x)))


def similarity_scores(mat, vec):
    half = len(vec)//2
    scores = []
    for row in mat:
        score = np.sum(row[:half] == vec[:half])
        score += np.sum(row[half:] == vec[half:]) * 2
        scores.append(score)
    return scores


def get_distribution(mat, vec, front=0.1):
    scores = similarity_scores(mat, vec)
    ind = np.argsort(scores[:-1])[::-1]
    front_ind = ind[:int(front*len(ind))]
    predictions = mat[front_ind+1][:, -1]
    return predictions


def get_prediction(mat, vec, front=0.1, adjust=True, prob=False):
    distribution = get_distribution(mat, vec, front)
    n = len(distribution)
    probs = np.ones(n) / n
    if adjust:
        half = n // 2
        probs[:half] *= 2
        probs[half:] = (1 - np.sum(probs[:half])) / (n-half)
    counts = distribution.reshape((1, -1)).dot(probs.reshape((-1, 1)))
    if not prob:
        prediction = 1 if counts >= 0.5 else 0
        return prediction
    else:
        return float(counts)


def find_similar_secu(code):
    industry = mapping[code]
    return mapping_reverse[industry]


# For every incoming day, update historical matrices of all security codes with current signs windows.
def update_all_signs_bit(date_):
    secu_updown = quote_test[quote_test.TradingDay == date_].loc[:, ['SecuCode', 'ChangePCT']]
    secu_updown.set_index(['SecuCode'], inplace=True)
    secu_signs = pd.DataFrame(secu_updown.ChangePCT.apply(lambda x: func(x)))
    secu_signs.columns = ['Sign']
    for code in secu_signs.index:
        print('Date:', date_, ' ', 'Code:', code)
        indust = mapping[code]
        update_sign = float(secu_signs.loc[code])
        signs_bit = signs_dict[indust][code]
        if len(signs_bit) == 0:
            continue
        last_row = signs_bit[-10:]
        last_row = np.array(list(last_row.to01())).astype('int')
        update_row = np.insert(last_row[1:], len(last_row)-1, update_sign)
        signs_bit.extend(update_row)
        signs_dict[indust][code] = signs_bit


# Given security probabilities and dates, generates weights for top N securities.
def generate_weights(secu_probs, date, top=10):
    key_li, val_li = [], []
    for key, val in secu_probs.items():
        key_li.append(key)
        val_li.append(val)
    codes_probs_df = pd.DataFrame([[date]*len(key_li), key_li, val_li]).T
    codes_probs_df.columns = ['tradingday', 'stockcode', 'stockweight']
    top_codes = codes_probs_df.sort_values(['stockweight'], ascending=False)[:top]
    prob_sum = top_codes.stockweight.sum()
    top_codes.stockweight = top_codes.stockweight.apply(lambda x: x/prob_sum)
    return top_codes


# Create a baseline by assigning each stock with the same weight.
def generate_baseline_weights(num_days=20, num_secu=15):
    base_weights = pd.DataFrame([0, 0, 0]).T
    base_weights.columns = ['tradingday', 'stockcode', 'stockweight']
    for i in range(num_days):
        date = str(quote_test.TradingDay.iloc[i+1])[:10]
        secu_codes = quote_test[quote_test.TradingDay == date].SecuCode.unique()
        secu_codes = np.random.permutation(secu_codes)
        np.random.seed(1)
        n = len(secu_codes[:num_secu])
        prob = np.ones(n) / n
        date_arr = np.array([date]*n)
        df = pd.DataFrame([date_arr, secu_codes[:num_secu], prob]).T
        df.columns = ['tradingday', 'stockcode', 'stockweight']
        base_weights = pd.concat([base_weights, df])
    base_weights = base_weights.iloc[1:]
    return base_weights


# Create a baseline by assigning same weights to stocks which performs well in the last three days. 
def generate_baseline_weights_2(num_days, num_secu=10):
    quote_3day = quote_sub[quote_sub.TradingDay.isin(['2017-08-10', '2017-08-09', '2017-08-08'])]
    ind = np.where(quote_3day.groupby(['SecuCode']).apply(lambda x: (x.ChangePCT > 0).all()))
    code_index = quote_3day.SecuCode.unique()
    secu_codes = code_index[ind]
    np.random.seed(1)
    base_weights = pd.DataFrame([0, 0, 0]).T
    base_weights.columns = ['tradingday', 'stockcode', 'stockweight']
    secu_codes = np.random.permutation(secu_codes)[:num_secu]
    for i in range(num_days):
        date = str(quote_test.TradingDay.iloc[i+1])[:10]
        current_secu_codes = quote_test[quote_test.TradingDay == date].SecuCode.unique()
        secu_codes = secu_codes[pd.Series(secu_codes).isin(current_secu_codes)]
        n = len(secu_codes)
        prob = np.ones(n) / n
        date_arr = np.array([date]*n)
        df = pd.DataFrame([date_arr, secu_codes[:num_secu], prob]).T
        df.columns = ['tradingday', 'stockcode', 'stockweight']
        base_weights = pd.concat([base_weights, df])
    base_weights = base_weights.iloc[1:]
    return base_weights



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base

TIME_SPACE = base.TIME_SPACE
STOCK_NUM = base.STOCK_NUM
HISTORY = base.HISTORY
ALL_HISTORY = base.ALL_HISTORY

# STOCKS=np.array(['东南网架','汉王科技'])
# STOCKS_PY=np.array(['DNWJ','HWKJ'])
# STOCKS=np.array(['南都电源' ,'棕榈股份' ,'以岭药业' ,'国元证券' ,'广深铁路' ,'吉林敖东' ,'光大证券' ,'安琪酵母', '江中药业' ,'光大银行'
#  ,'南山铝业', '江西铜业' '盐湖股份' ,'浦发银行' ,'中国重工'])

# STOCKS=np.array(['招商银行','中信证券','国元证券','平安银行','东方财富','吉林敖东','兴业银行','光大证券','光大银行', '亚泰集团'
#  ,'交通银行' ,'辽宁成大' ,'华泰证券' ,'中国银行'])

STOCKS=np.array(['南都电源' ,'棕榈股份' ,'以岭药业' ,'国元证券' ,'交通银行' ,'辽宁成大' ,'华泰证券' ,'中国银行'])

# STOCKS_PY=np.array(['HWKJ'])

def load_data(path):
    # 'SecuCode','SecuAbbr','TradingDay','Close','High','Low','Avg','ChangePCT','TurnoverVolume','TurnoverValue'
    df = pd.read_csv(path)
    for i in range(STOCKS.size):
        # .isin(stocks)
        stock = df[df['SecuAbbr'] == STOCKS[i]][["Close", "High", "Low"]]
        if i ==0:
            X=stock.values
        else:
            X=np.dstack([X,stock.values])

    #[ALL_HISTORY, 3, STOCK_NUM]
    return X.reshape((-1, STOCK_NUM, 3))

def draw_stocks_price_plot(path):
    df = pd.read_csv(path)
    for i in range(STOCKS.size):
        stock = df[df['SecuAbbr'] == STOCKS[i]]
        stock["TradingDay"] = stock["TradingDay"].apply(lambda x: pd.to_datetime(x).date());
        # plt.plot(stock["TradingDay"],stock["Close"], '.', label=STOCKS[i], markersize=1)
        plt.plot(stock["TradingDay"], stock["Close"], '-', label=STOCKS[i], linewidth=1)

    plt.xlabel('date')
    plt.ylabel('price')
    # plt.legend(STOCKS_PY,loc='upper right')
    # plt.tick_params(axis='both', which='major', labelsize=5)
    plt.xticks(fontsize=8, rotation=45)
    plt.show()


def draw_reward_ratio_plot(path, r):
    df = pd.read_csv(path)
    stock = df[df['SecuAbbr'] == STOCKS[0]]
    stock["TradingDay"] = stock["TradingDay"].apply(lambda x: pd.to_datetime(x).date())

    longtimeR = np.empty([HISTORY, 1])
    for i in range(HISTORY):
        longtimeR[i] = np.prod(r[:i+1]) - 1

    plt.plot(stock["TradingDay"].iloc[:HISTORY], longtimeR, '-', label='long_time', linewidth=1)
    plt.xlabel('date')
    plt.ylabel('long-time-ratio')
    plt.xticks(fontsize=8, rotation=45)
    plt.show()

draw_stocks_price_plot(base.QUTOE_PATH)
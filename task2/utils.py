import base
import pandas as pd

def read_records_csv():
    df = base.load_records_csv()
    print(df['PortCode'].unique().size)

# 选盈利top_n的策略
def top_nav_portcode(n):
    df = base.load_nav_csv()
    nav = df.groupby(['PortCode'])['Nav'].apply(lambda x: x.iloc[-1]).reset_index(name='Q')
    nav = nav.sort_values(by=['Q'], ascending=False)
    names = nav.head(n)['PortCode'].tolist()

    return names

#查看portcode的nav记录
def check_portcode_detial(portcode):
    df = base.load_nav_csv()
    df = df[df['PortCode'] == portcode]
    return df

#查看多个portcode的nav记录
def check_portcodes_detial(portcodes):
    df = base.load_nav_csv()
    tmp1 = df[df['PortCode'] == portcodes[0]][['NavDate', 'Nav']]
    for p in portcodes[1:]:
        tmp2 = df[df['PortCode'] == p][['NavDate', 'Nav']]
        tmp1 = pd.merge(tmp1, tmp2, on=['NavDate'], how='left')
    tmp1.reset_index(['NavDate'])
    portcodes.insert(0,'NavDate')
    tmp1.columns = portcodes
    return tmp1

# codes = top_nav_portcode(3)
df = check_portcodes_detial(['ZH000199', 'ZH000283'])
print(df)

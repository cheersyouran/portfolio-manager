import base

def read_records_csv():
    df = base.load_records_csv()
    print(df['PortCode'].unique().size)


def top_nav_portcode(n):
    # 选盈利top_n的策略
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

codes = top_nav_portcode(3)
print(codes)

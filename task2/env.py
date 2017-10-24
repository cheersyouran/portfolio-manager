import pandas as pd
import numpy as np
import base

nav = base.load_nav_csv()
nav.groupby(['PortCode'])


s1 = base.load_strategy1_data()
s1 = s1.drop(['SecuAbbr', 'label'], axis=1)
s1.reset_index(['TradingDay']).groupby(['SecuCode'])





import base
from datetime import datetime,timedelta

time = datetime(2015, 11, 3)

class Market():
    def __init__(self):
        self.insdustry_quote = base.load_industryquote_xlsx()
        self.nav = base.load_nav_csv()
        self.records = base.load_records_csv()
        self.current_date = time
        self.trading_data_df = base.load_trading_day_csv()

    # return : industry_quote, records, nav
    def get_daily_market(self, date=None):
        if(date == None):
            date = self.current_date
        industry_quote = self.insdustry_quote[self.insdustry_quote['TradingDay'] == date]
        records = self.records[self.records['Updated'].apply(lambda x : x.date()) == date]
        nav = self.nav[self.nav['NavDate'].apply(lambda x : x.date()) == date]
        return industry_quote, records, nav

    def get_past_market(self, date=None):
        if (date == None):
            date = self.current_date
        industry_quote = self.insdustry_quote[self.insdustry_quote['TradingDay'] <= date]
        records = self.records[self.records['Updated'] <= date]
        nav = self.nav[self.nav['NavDate'] <= date]
        return industry_quote, records, nav

    def get_previous_market(self, date=None):
        if (date == None):
            date = self.current_date
        industry_quote = self.insdustry_quote[self.insdustry_quote['TradingDay'] >= date]
        records = self.records[self.records['Updated'] >= date]
        nav = self.nav[self.nav['NavDate'] >= date]
        return industry_quote, records, nav

    def get_between_market(self, from_date=None, to_date=None):
        if (from_date == None):
            from_date = self.current_date
        if (to_date == None):
            to_date = self.current_date
        industry_quote = self.insdustry_quote[(self.insdustry_quote['TradingDay'] >= from_date) & (self.insdustry_quote['TradingDay'] <= to_date)]
        records = self.records[(self.records['Updated'] >= from_date) & (self.records['Updated'] <= to_date)]
        nav = self.nav[(self.nav['NavDate'] >= from_date) & (self.nav['NavDate'] <= to_date)]
        return industry_quote, records, nav

    def pass_a_day(self):
        next_day = self.trading_data_df.iloc[self.trading_data_df[(self.trading_data_df['TradingDate'] == date)].index+1].values
        self.current_date = next_day[0]

    def reset(self):
        self.current_date = time

if __name__ == '__main__':
    m = Market()
    print(m.get_daily_market())
    print(m.get_past_market())
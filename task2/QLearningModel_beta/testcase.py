import base
from State_Space.Update_IR_rank import Update_IR_rank

def test_Update_IR_rank():
    records = base.load_records_csv()
    industry_quote = base.load_industryquote_xlsx()
    nav = base.load_nav_csv()
    quote = base.load_quote_csv()
    IR_rank = base.load_irweek_csv()

    date = '2017-01-06'
    df_records = records[records.Updated <= date]
    df_ind_quote = industry_quote[industry_quote.TradingDay <= date]
    df_nav = nav[nav.NavDate <= date]
    df_quote = quote[quote.TradingDay <= date]

    Update_IR_rank(date, df_records, df_ind_quote, df_nav, df_quote, IR_rank)

if __name__ == '__main__':
    test_Update_IR_rank()
import alpaca_trade_api as tradeapi
from conf.config import *

def connect_to_api():
    base_url = 'https://paper-api.alpaca.markets'
    api_key_id = API_KEY
    api_secret = SECRET_KEY

    api = tradeapi.REST(
        base_url=base_url,
        key_id=api_key_id,
        secret_key=api_secret
    )
    return(api)

def get_active_assets(api):
    # Get a list of all active assets.
    active_assets = api.list_assets(status='active')
    # Filter the assets down to just those on NASDAQ.
    nasdaq_assets = [a for a in active_assets if a.exchange == 'NASDAQ']
    print(nasdaq_assets)

def is_tradable(api, symbol):
    # Check if AAPL is tradable on the Alpaca platform.
    aapl_asset = api.get_asset(symbol)
    if aapl_asset.tradable:
        print('We can trade ' + symbol + '.')

def check_market_hours(api, date):
    # Check if the market is open now.
    clock = api.get_clock()
    print('The market is {}'.format('open.' if clock.is_open else 'closed.'))
    calendar = api.get_calendar(start=date, end=date)[0]
    print('The market opened at {} and closed at {} on {}.'.format(
        calendar.open,
        calendar.close,
        date
    ))

def submit_market_order(api, symbol, qty, side):
    # Submit a market order to buy 1 share of Apple at market price
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='market',
        time_in_force='gtc'
    )

def submit_limit_order(api, symbol, qty, side, limit_price):
    # Submit a limit order to attempt to sell 1 share of AMD at a
    # particular price ($20.50) when the market opens
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='limit',
        time_in_force='opg',
        limit_price=limit_price
    )

def get_position(api, symbol):
    return(api.get_position(symbol))

def list_positions(api):
    # Get a list of all of our positions.
    return(api.list_positions())

def trade(api, symbol):
    pass

def main():
    api = connect_to_api()
    get_active_assets(api)
    is_tradable(api, "AAPL")
    check_market_hours(api, '2020-12-01')

if __name__ == "__main__":
    main()

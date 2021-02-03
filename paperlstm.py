import alpaca_trade_api as tradeapi
from conf.config import *
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
pd.options.mode.chained_assignment = None

training_data_len = 0

def prepare_data(symbol):
    df = web.DataReader(symbol, data_source='yahoo', start='2012-01-01', end='2020-01-01')
    print(df)
    #capture and split dataset
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    #normalize all the data to be between 0 and 1 to account for price differences
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    #create scaled training dataset
    train_data = scaled_data[0:training_data_len, :]
    #split into x and y using past 60 days
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])

    #convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #reshape data so it is 3 dimensional
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #create scaled testing dataset
    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return(x_train, y_train, x_test, scaler, data, training_data_len)


def train(x_train, y_train):
    #build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return(model)

def predict(x_test, model, scaler, data, training_data_len):
    plt.style.use('fivethirtyeight')
    pd.options.mode.chained_assignment = None
    # get models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    '''
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    '''
    return(scaler, model)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def trade(scaler, model, data):
    shares = 0
    money = 10000
    data = data[-366:]
    data = data.values.tolist()

    in_trade = False
    down = False
    up = False

    start_date = date(2018, 7, 20)
    end_date = date(2018, 10, 1)
    i = 0
    for single_date in daterange(start_date, end_date):
        #get date
        todays_date = single_date.strftime("%Y-%m-%d")
        #predict tommorrows price
        df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end=todays_date)
        df2 = df.filter(['Close'])
        df2 = df2[-60:]
        last_60_days = df2.values
        last_60_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        #tommorrows predicted price
        pred = pred_price[0][0]
        print("prediction")
        print(pred)
        #todays price
        today = data[i][0]
        print("today")
        print(today)
        print("")
        if(today > pred):
            down = True
        else:
            up = True

        if(in_trade):
            if(down):
                #sell
                print("sell")
                money = shares * today
                print("money")
                print(money)
                print("")
                shares = 0
                in_trade = False
        else:
            if(up):
                #buy
                print("buy")
                shares = money / today
                print("shares")
                print(shares)
                print("")
                money = 0
                in_trade = True

        down = False
        up = False
        i += 1

    #cash out
    if(in_trade):
        #sell
        money = shares * today
        shares = 0
        in_trade = False

    print("results:")
    print("starting money")
    print(10000)
    print("ending money")
    print(money)
    print("profit")
    print(money - 10000)

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
        return True

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

def do_trade(api, symbol):
    dat = prepare_data(symbol)
    model = train(dat[0], dat[1])
    prediction = predict(dat[2], model, dat[3], dat[4], dat[5])
    trade(prediction[0], prediction[1], prediction[4])

def main():
    api = connect_to_api()
    if (is_tradable(api, "TWTR")):
        do_trade(api, "TWTR")

if __name__ == "__main__":
    main()

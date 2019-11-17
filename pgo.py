
import requests
import pandas as pd
import talib as ta
import datetime
import math
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from pandas import Timestamp

class PgoTrans():
    def __init__(self):
        self.exchange_url = 'https://api.binance.com'
        self.api_point = '/api/v1/klines'

    def get_binance_candle(self):
        url = self.exchange_url + self.api_point
        res = requests.get(url + '?symbol=BTCUSDT&interval=12h&limit=1000')
        data_json = res.json()
        headers = ('Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Tr')
        df = pd.DataFrame(columns=headers)

        for i in range(1, len(data_json)):
            data_item = data_json[i]
            pre_data_item = data_json[i-1]
            df.loc[i] = (datetime.datetime.utcfromtimestamp(data_item[0]/1000.0).strftime('%Y-%m-%d'), float(data_item[1]), float(data_item[2])
                         , float(data_item[3]), float(data_item[4]), float(data_item[5]),
                         max(float(data_item[2])-float(data_item[3]), math.fabs(float(data_item[2])-float(pre_data_item[4])),
                             math.fabs(float(data_item[3]) - float(pre_data_item[4]))))

        df['Date'] = pd.to_datetime(df['Date'])
        df["Date"] = df["Date"].apply(mdates.date2num)

        print(df.tail(5))
        print(ta.get_function_groups())
        ohlc = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
        ohlc['EMA_32'] = ta.EMA(df.Close.values, timeperiod=89)
        ohlc['EMA_32_1'] = ta.EMA(df.Tr.values, timeperiod=89)
        ohlc['SMA_32'] = ta.SMA(ohlc.Close.values, timeperiod=89)
        ohlc['PGO_32'] = (ohlc['Close'] - ohlc['SMA_32']) / ohlc['EMA_32_1']
        f, axarr = plt.subplots(2, sharex=True, sharey=False)
        # plot the candlesticks
        axarr[0].set_title('CandleStick')
        candlestick_ohlc(axarr[0], ohlc.values, width=0.6, colorup='green', colordown='red')
        axarr[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axarr[0].xaxis.set_major_locator(ticker.MaxNLocator(6))
        axarr[0].grid(True)
        axarr[1].set_title('PGO')
        pgo_data = ohlc.PGO_32.values.copy()
        pgo_buy_data = ohlc.PGO_32.values.copy()
        pgo_sell_data = ohlc.PGO_32.values.copy()
        normal_data = ohlc.PGO_32.values.copy()
        for idx, j in np.ndenumerate(pgo_data):
            if j >= 3:
                pgo_buy_data[idx] = j
                pgo_sell_data[idx] = np.nan
                normal_data[idx] = j
            elif j <= -3:
                pgo_sell_data[idx] = j
                pgo_buy_data[idx] = np.nan
                normal_data[idx] = j
            else:
                normal_data[idx] = j
                pgo_sell_data[idx] = np.nan
                pgo_buy_data[idx] = np.nan
        axarr[1].plot(ohlc['Date'], normal_data, color='blue', label='PGO_32')
        axarr[1].plot(ohlc['Date'], pgo_buy_data, color='green', label='PGO_32')
        axarr[1].plot(ohlc['Date'], pgo_sell_data, color='red', label='PGO_32')
        axarr[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axarr[1].xaxis.set_major_locator(ticker.MaxNLocator(6))
        axarr[1].text(df['Date'].iloc[0], 8, 'length=input(89)\npgo = (close - sma(close, length))/ema(tr, length)', color='green',
                      fontsize=15)
        axarr[1].grid(True)
        print(ohlc.tail(5))
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
        plt.show()



trans = PgoTrans()
trans.get_binance_candle()
pgo.py

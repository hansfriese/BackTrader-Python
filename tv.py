import pandas_datareader.data as web
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import backtrader as bt
import numpy as np
from backtrader.feeds import PandasData
from matplotlib import interactive

symbol = 'AMZN'
upper_percent = None
lower_percent = None
upper_band = None
lower_band = None

shortLag = 0.4
longLag = 0.8
topLookBackPeriod = 200
bottomLookBackPeriod = 200
ShowThresholdLine = True
ShowWarningThresholdLine = True
orderThreshold = 90

def ReadHistoryData():
    start = '2015-09-5'
    end = '2019-05-08'

    df = web.DataReader(name=symbol, data_source='iex', start=start, end=end)
    df.to_csv('data.csv'.format(symbol))


def CalcIndicator():

    global upper_percent
    global lower_percent

    df_all = pd.read_csv('data.csv'.format(symbol), index_col='date',
                         parse_dates=True, usecols=['date', 'open', 'high', 'low', 'close'],
                         na_values='nan')

    midPrice = (df_all['high'] + df_all['low']) / 2
    length = len(midPrice)
    g = 0

    L0 = np.empty(length)
    L1 = np.empty(length)
    L2 = np.empty(length)
    L3 = np.empty(length)
    shortfinal = np.empty(length)
    longfinal = np.empty(length)
    upper_val = np.empty(length)
    lower_val = np.empty(length)
    upper_percent = np.empty(length)
    lower_percent = np.empty(length)

    for k in range(2):
        if k == 0:
            g = shortLag
        else:
            g = longLag

        for i in range(length):
            L0[i] = (1 - g) * midPrice[i]
            tmpVal = 0
            if i > 0:
                tmpVal = g * L0[i-1]
            L0[i] += tmpVal

            L1[i] = -g * L0[i]
            tmpVal = 0
            if i > 0:
                tmpVal = L0[i-1] + g * L1[i-1]
            L1[i] += tmpVal

            L2[i] = -g * L1[i]
            tmpVal = 0
            if i > 0:
                tmpVal = L1[i-1] + g * L2[i-1]
            L2[i] += tmpVal

            L3[i] = -g * L2[i]
            if i > 0:
                tmpVal = L2[i-1] + g * L3[i-1]
            L3[i] += tmpVal

            if k == 0:
                shortfinal[i] = (L0[i] + 2 * L1[i] + 2 * L2[i] + L3[i]) / 6
            else:
                longfinal[i] = (L0[i] + 2 * L1[i] + 2 * L2[i] + L3[i]) / 6

    for i in range(length):
        upper_val[i] = (shortfinal[i] - longfinal[i]) / longfinal[i] * 100
        lower_val[i] = (longfinal[i] - shortfinal[i]) / longfinal[i] * 100
        lessNum = 0
        totNum = 0
        for k in range(i - topLookBackPeriod, i):
            if k < 0:
                continue
            totNum += 1
            if upper_val[k] <= upper_val[i]:
                lessNum += 1
        if totNum > 0:
            upper_percent[i] = 100.0 * lessNum / totNum
        else:
            upper_percent[i] = 0

        totNum = 0
        lessNum = 0
        for k in range(i - bottomLookBackPeriod, i):
            if k < 0:
                continue
            totNum += 1
            if lower_val[k] <= lower_val[i]:
                lessNum += 1
        if totNum > 0:                
            lower_percent[i] = -100.0 * lessNum / totNum
        else:
            lower_percent[i] = 0


def DrawIndicator():
    global upper_band
    global lower_band

    df = pd.read_csv('data.csv'.format(symbol), index_col='date',
                     parse_dates=True, usecols=['date', 'close'],
                     na_values='nan')

    df = df.rename(columns={'close': symbol})
    df.dropna(inplace=True)

    tmp = df.rolling(window=2).mean()

    upper_band = tmp
    upper_band = upper_band.rename(columns={symbol: 'RankT'})
    lower_band = tmp
    lower_band = lower_band.rename(columns={symbol: 'RankB'})

    for i in range(len(upper_percent)):
        upper_band['RankT'][i] = upper_percent[i]
        lower_band['RankB'][i] = lower_percent[i]

    _, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={
                           'wspace': 0, 'hspace': 0.03})

    ax = df.plot(title='{} Chart and Indicator'.format(
        symbol), ax=axes[0])
    ax.get_xaxis().set_visible(False)
    ax.set_facecolor('black')
    # ax.tick_params(labelbottom=False, bottom=False, )
    ax.set_ylabel('Price')
    ax.grid()
    interactive(True)

    df1 = df.join(upper_band).join(lower_band)
    df1 = df1.drop(symbol, axis=1)
    ax1 = df1.plot(ax=axes[1], linewidth=0)
    ax1.set_facecolor('black')
    ax1.fill_between(
        df.index, lower_band['RankB'], 0, color='#7D7D7D')
    ax1.fill_between(
        df.index, 0, upper_band['RankT'], color='#535353')
    ax1.fill_between(
        df.index, 0, upper_band['RankT'], color='#A65200', where=upper_band['RankT'] > 70)
    ax1.fill_between(
        df.index, 0, upper_band['RankT'], color='#A60000', where=upper_band['RankT'] > 90)
    ax1.fill_between(
        df.index, 0, lower_band['RankB'], color='#005300', where=lower_band['RankB'] < -70)
    ax1.fill_between(
        df.index, 0, lower_band['RankB'], color='#00A600', where=lower_band['RankB'] < -90)

    if ShowThresholdLine:
        plt.axhline(y=90, color='#A60000', linestyle='-')
        plt.axhline(y=-90, color='#00A600', linestyle='-')
    if ShowWarningThresholdLine:
        plt.axhline(y=70, color='#A65200', linestyle='-')
        plt.axhline(y=-70, color='#005300', linestyle='-')
    plt.show()
    interactive(False)


class MyStrategy(bt.Strategy):
    def __init__(self):
        self.order = None
        self.typo = None

    def next(self):
        if self.datas[0].RankT[0] > orderThreshold and self.typo != 1:
            self.order = self.buy()
            self.typo = 1
        if self.datas[0].RankB[0] < -orderThreshold and self.typo != 0:
            self.order = self.sell()
            self.typo = 0


class PandasData_Signal(PandasData):
    lines = ('RankT', 'RankB')
    params = (('RankT', -1), ('RankB', -1))
    datafields = PandasData.datafields + (['RankT', 'RankB'])


def BackTest():
    global symbol

    cerebro = bt.Cerebro()
    dataframe = pd.read_csv('data.csv',
                            skiprows=0,
                            header=0,
                            parse_dates=True,
                            index_col=0)
    dataframe = dataframe.join(upper_band).join(lower_band)
    data = PandasData_Signal(dataname=dataframe,
                             datetime=None,
                             open=0,
                             high=1,
                             low=2,
                             close=3,
                             volume=4,
                             RankT=5,
                             RankB=6
                             )

    cerebro.broker.setcash(10000.0)
    cerebro.adddata(data)
    cerebro.addstrategy(MyStrategy)
    cerebro.run()
    cerebro.plot()


ReadHistoryData()
CalcIndicator()
DrawIndicator()
BackTest()

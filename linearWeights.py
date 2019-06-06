import datetime as dt
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import featurize
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import warnings; warnings.simplefilter('ignore')
import util
import random
random.seed(42)

def output(start, end, ticker, period, model):
    df = data.DataReader(ticker, 'yahoo', start, end)
    raw = featurize.featurize(df)
    raw = raw.dropna()
    x, y = util.prepare(raw, period)
    toDrop = ['Volume', 'VolumeChange', 'High250', 'Low250', '%D',
              'TRIXSignal', 'Mov5', 'Mov20', 'Mov60', 'Mov250']
    x = x.drop(columns=toDrop)
    
    ((xTest, xTrain), (yTest, yTrain)) = util.split(x, y, 20)
    lin = model.fit(xTrain, yTrain)

    return lin.coef_

def getAverage(ticker, period, model):
    months = random.sample(range(1, 13), 5)
    trials = [output(dt.datetime(2013+i, months[i], 1), dt.datetime(2015+i, months[i], 1), 
                     ticker, period, model) for i in range(5)]
    df = pd.DataFrame(trials)
    return df.mean(axis=0)

def line(ticker, model):
    fiveDay = getAverage(ticker, 5, model)
    tenDay = getAverage(ticker, 10, model)
    twentyDay = getAverage(ticker, 20, model)
    for period in [fiveDay, tenDay, twentyDay]:
        plt.figure()
        plt.ylabel('Average Feature Weight')
        plt.bar(['Momentum', 'CloseChange', 'MomentumChange', 'MACD', 'Signal',
       'MACD-Signal', 'RSI', '%K', '%K-%D', '%MFI', 'Capacity1',
       'Capacity2', 'AccDist', 'AccDistema21', 'PVT', 'PVTema21', 'ChOsc',
       'adx', 'di_plus', 'di_minus', 'DIdiff', 'TRIX', 'TRIX-TRIXSignal',
       'WillAD', 'WillR'], period)
        plt.xticks(rotation='vertical')
        plt.tight_layout()

for ticker in ['PFE']:#, 'PFE', 'JNJ', 'CVX', 'XOM', 'PG']:
    line(ticker, Ridge(alpha=0.1, max_iter=1000))
import datetime as dt
import pandas as pd
from pandas_datareader import data
import featurize
import statsmodels.api as sm
import util
import warnings; warnings.simplefilter('ignore')
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
    X2 = sm.add_constant(xTrain)
    est = sm.OLS(yTrain.reset_index().drop(columns=['Date']).astype(float), 
                 X2.reset_index().drop(columns=['Date']).astype(float))
    est2 = est.fit()
    pvalues = (est2.pvalues)

    return pvalues

def getAverage(ticker, period, model):
    months = random.sample(range(1, 13), 5)
    trials = [output(dt.datetime(2013+i, months[i], 1), dt.datetime(2015+i, months[i], 1), ticker, period, model) for i in range(5)]
    df = pd.DataFrame(trials)
    return df.apply(lambda row: row.between(left=0, right=0.05).sum(), axis=0)
    #return df.mean(axis=0)

def line(ticker, model):
    fiveDay = getAverage(ticker, 5, model)
    tenDay = getAverage(ticker, 10, model)
    twentyDay = getAverage(ticker, 20, model)
    toPrint = '{:<15} & {} & {} & {}'
    #toPrint = '{:<15} & {:.5f} & {:.5f} & {:.5f}'
    for i in fiveDay.index:
        print(toPrint.format(i, fiveDay[i], tenDay[i], twentyDay[i]))
    #print(toPrint.format(ticker, fiveDay[0], fiveDay[1], tenDay[0], tenDay[1], twentyDay[0], twentyDay[1]))

for ticker in ['MRK']:#, 'PFE', 'JNJ', 'CVX', 'XOM', 'PG']:
    line(ticker, None)

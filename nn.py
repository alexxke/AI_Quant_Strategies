import datetime as dt
from pandas_datareader import data
import featurize
import keras
import warnings; warnings.simplefilter('ignore')
import random
random.seed(42)
import acc
import util

def output(start, end, ticker, period, model):
    df = data.DataReader(ticker, 'yahoo', start, end)
    raw = featurize.featurize(df)
    raw = raw.dropna()
    
    x, y = util.prepare(raw, period)
    toDrop = ['Volume', 'VolumeChange', 'High250', 'Low250', '%D',
              'TRIXSignal', 'Mov5', 'Mov20', 'Mov60', 'Mov250']
    x = x.drop(columns=toDrop)
    ((xTest, xTrain), (yTest, yTrain)) = util.split(x, y, 20)

    es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=50)
    model.fit(xTrain.values, yTrain.values, verbose=0, epochs=2000, callbacks=[es])

    return (acc.accuracy(df, xTrain, yTrain, model.predict(xTrain.values)), 
            acc.accuracy(df, xTest, yTest, model.predict(xTest.values)))

def getAverage(ticker, period, model):
    months = random.sample(range(1, 13), 5)
    trials = [output(dt.datetime(2013+i, months[i], 1), dt.datetime(2015+i, months[i], 1), 
                     ticker, period, model) for i in range(5)]
    return [sum(i) / len(i) for i in zip(*trials)]

def line(ticker, model):
    fiveDay = getAverage(ticker, 5, model)
    tenDay = getAverage(ticker, 10, model)
    twentyDay = getAverage(ticker, 20, model)
    toPrint = '{} & {:.5f} & {:.2f} & {:.5f} & {:.2f} & {:.5f} & {:.2f} \\\\'
    print(toPrint.format(ticker, fiveDay[0], fiveDay[1], tenDay[0], tenDay[1], twentyDay[0], twentyDay[1]))

for ticker in ['MRK', 'PFE', 'JNJ', 'CVX', 'XOM', 'PG']:
    # try to add some dropout or L2 penalty to prevent overfitting
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, activation='tanh'))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    line(ticker, model)
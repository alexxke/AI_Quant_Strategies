import numpy as np

# random strategy
def random(df, x, y):
    ctr = 0
    for close, actual in zip(df['Close'][x.index], y):#, lin.predict(x)):
        result = np.random.randint(2)
        if result == 1: ctr += 1
    return ctr / len(x.index)

# always down strategy
def pessimist(df, x, y):
    ctr = 0
    for close, actual in zip(df['Close'][x.index], y):
        result = False
        if actual < close:
            result = True
        if result: ctr += 1
    return ctr / len(x.index)

# always up strategy
def optimist(df, x, y):
    ctr = 0
    for close, actual in zip(df['Close'][x.index], y):
        result = False
        if actual > close:
            result = True
        if result: ctr += 1
    return ctr / len(x.index)

# dollar amount if you execute the trades
def profit(df, x, y, yHat):
    dollars = 0
    for close, actual, predict in zip(df['Close'][x.index], y, yHat):
        if (predict < close and actual < close) or (predict > close and actual > close):
            dollars += abs(close - actual)
    return dollars

# percentage of stock directions you predicted right
def accuracy(df, x, y, yHat):
    ctr = 0
    for close, actual, predict in zip(df['Close'][x.index], y, yHat):
        result = False
        if (predict < close and actual < close) or (predict > close and actual > close):
            result = True
        if result: ctr += 1
        #print(close, actual, predict, result)
    return ctr / len(x.index)

# percentage of stock directions predicted right with binary labels
def log(df, x, y, yHat):
    ctr = 0
    for (day, close), actual, predict in zip(df['Close'][x.index].iteritems(), y, yHat):
        ground = 0
        if close <= actual:
            ground = 1
        if ground == predict: ctr += 1
    return ctr / len(x.index)
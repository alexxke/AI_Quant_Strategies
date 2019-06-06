from sklearn.linear_model import LinearRegression
import pandas as pd

# normalize, set the prediction period, drop nas
def prepare(rawFeatures, period=5):
    normalizedX = (rawFeatures - rawFeatures.min()) / rawFeatures.std()
    normalizedX = normalizedX.iloc[:-period]
    # base prediction only on technical indicators
    normalizedX = normalizedX.drop(columns=['Close'])
    y = rawFeatures['Close'][period:]
    return (normalizedX, y)

# maybe try splitting a test set by choosing random days out of the regression
def split(x, y, numTest):
    yTest = y[-numTest:]
    yTrain = y[:-numTest]
    xTest = x[-numTest:]
    xTrain = x[:-numTest]
    return ((xTest, xTrain), (yTest, yTrain))

def forwardFeatureSelection(xTrain, yTrain, maxFeatures, evaluate, threshold=-1):
    features = list()
    prev = 0
    for _ in range(maxFeatures):
        possible = list()
        model = LinearRegression()
        for feature in xTrain.columns:
            if feature not in features:
                cur = features.copy()
                cur.append(feature)
                model.fit(xTrain[cur], yTrain)
                possible.append((evaluate(model, xTrain[cur], yTrain), feature))
        possible.sort(key=lambda x : x[0], reverse = True)
        if possible[0][0] - prev <= threshold: break
        prev = possible[0][0]
        features.append(possible[0][1])
    return features

def makeBinary(df, x, y):
    binary = pd.Series()
    for (day, close), actual in zip(df['Close'][x.index].iteritems(), y):
        if close <= actual:
            binary[day] = 1
        else:
            binary[day] = 0
    return binary
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap(x):
    corr = x.corr()
    plt.figure()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

def weightFigure(featureNames, weights):
    plt.figure()
    plt.bar(featureNames, weights)
    plt.xticks(rotation='vertical')
    plt.tight_layout()

def predictCurve(xTest, yTest, yHat):
    plt.figure()
    plt.plot(xTest.index, yTest, label='Target')
    plt.plot(xTest.index, yHat, label='Prediction')
    plt.legend()
    plt.ylabel('Share Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
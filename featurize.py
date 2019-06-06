import numpy as np
import pandas as pd
import indicators

def featurize(df):
    def MACD(series):
        ema26 = pd.DataFrame.ewm(series, span=26).mean()
        ema12 = pd.DataFrame.ewm(series, span=12).mean()
        macd = ema12 - ema26
        return macd
    def RSI(series, period):
        delta = series.diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
        u = u.drop(u.index[:(period-1)])
        d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
        d = d.drop(d.index[:(period-1)])
        rs = pd.DataFrame.ewm(u, com=period-1, adjust=False).mean() \
        / pd.DataFrame.ewm(d, com=period-1, adjust=False).mean()
        return 100 - 100 / (1 + rs)
    def STOK(close, low, high, n): 
        STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
        return STOK
    def STOD(close, low, high, n):
        stok = STOK(close, low, high, n)
        STOD = stok.rolling(3).mean()
        return STOD

    rawFeatures = df.loc[:, ['Close', 'Volume']]
    rawFeatures['Momentum'] = df['Close'] * df['Volume']
    rawFeatures['CloseChange'] = df['Close'].diff() / df['Close']
    rawFeatures['VolumeChange'] = df['Volume'].diff() / df['Volume']
    rawFeatures['MomentumChange'] = rawFeatures['Momentum'].diff() / rawFeatures['Momentum']
    rawFeatures['MACD'] = MACD(df['Close'])
    rawFeatures['Signal'] = rawFeatures['MACD'].ewm(com=9, adjust=True).mean()
    rawFeatures['MACD-Signal'] = rawFeatures['MACD'] - rawFeatures['Signal']
    rawFeatures['RSI'] = RSI(df['Close'], 14) # make sure to drop rows with nan RSI values
    rawFeatures['%K'] = STOK(df['Close'], df['Low'], df['High'], 14) # stochastic oscillator
    rawFeatures['%D'] = STOD(df['Close'], df['Low'], df['High'], 14)
    rawFeatures['%K-%D'] = rawFeatures['%K'] - rawFeatures['%D']
    
    temp = indicators.money_flow_index(df)
    temp = temp.set_index('Date')
    rawFeatures['%MFI'] = 100 * temp['money_flow_index']
    
    rawFeatures['High250'] = df['Close'].rolling(250).max()
    rawFeatures['Low250'] = df['Close'].rolling(250).min()
    rawFeatures['Capacity1'] = rawFeatures['Close'] / rawFeatures['High250']
    rawFeatures['Capacity2'] = rawFeatures['Close'] / (rawFeatures['High250'] - rawFeatures['Low250'])
    
    rawFeatures['Mov5'] = pd.DataFrame.ewm(rawFeatures['Close'], span=5).mean()
    rawFeatures['Mov20'] = pd.DataFrame.ewm(rawFeatures['Close'], span=20).mean()
    rawFeatures['Mov60'] = pd.DataFrame.ewm(rawFeatures['Close'], span=60).mean()
    rawFeatures['Mov250'] = pd.DataFrame.ewm(rawFeatures['Close'], span=250).mean()
    
    temp = indicators.acc_dist(df)
    rawFeatures = rawFeatures.join(temp[['AccDist', 'AccDistema21']])
    temp = indicators.price_volume_trend(df)
    rawFeatures = rawFeatures.join(temp[['PVT', 'PVTema21']])
    
    temp = indicators.chaikin_oscillator(df)
    rawFeatures = rawFeatures.join(temp['ChOsc'])
    temp = indicators.directional_movement_index(df)
    rawFeatures = rawFeatures.join(temp[['adx', 'di_plus', 'di_minus']])
    rawFeatures['DIdiff'] = rawFeatures['di_plus'] - rawFeatures['di_minus']
    temp = indicators.trix(df)
    rawFeatures = rawFeatures.join(temp[['TRIX', 'TRIXSignal']])
    rawFeatures['TRIX-TRIXSignal'] = rawFeatures['TRIX'] - rawFeatures['TRIXSignal']
    temp = indicators.williams_ad(df)
    rawFeatures = rawFeatures.join(temp['WillAD'])
    temp = indicators.williams_r(df)
    rawFeatures = rawFeatures.join(temp['WillR'])
    return rawFeatures
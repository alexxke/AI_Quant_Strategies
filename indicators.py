import numpy as np
import pandas as pd

def acc_dist(data, trend_periods=21, open_col='Open', high_col='High', low_col='Low', close_col='Close', vol_col='Volume'):
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            ac = 0
        data.set_value(index, 'AccDist', ac)
    data['AccDistema' + str(trend_periods)] = data['AccDist'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    
    return data

def price_volume_trend(data, trend_periods=21, close_col='Close', vol_col='Volume'):
    data = data.reset_index()
    for index, row in data.iterrows():
        if index > 0:
            last_val = data.at[index - 1, 'PVT']
            last_close = data.at[index - 1, close_col]
            today_close = row[close_col]
            today_vol = row[vol_col]
            current_val = last_val + (today_vol * (today_close - last_close) / last_close)
        else:
            current_val = row[vol_col]

        data.set_value(index, 'PVT', current_val)

    data['PVTema' + str(trend_periods)] = data['PVT'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    data = data.set_index('Date')
    return data

def average_true_range(data, trend_periods=14, open_col='Open', high_col='High', low_col='Low', close_col='Close', drop_tr = True):
    for index, row in data.iterrows():
        prices = [row[high_col], row[low_col], row[close_col], row[open_col]]
        if index > 0:
            val1 = np.amax(prices) - np.amin(prices)
            val2 = abs(np.amax(prices) - data.at[index - 1, close_col])
            val3 = abs(np.amin(prices) - data.at[index - 1, close_col])
            true_range = np.amax([val1, val2, val3])

        else:
            true_range = np.amax(prices) - np.amin(prices)

        data.set_value(index, 'true_range', true_range)
    data['atr'] = data['true_range'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    if drop_tr:
        data = data.drop(['true_range'], axis=1)
        
    return data

def chaikin_oscillator(data, periods_short=3, periods_long=10, high_col='High',
                       low_col='Low', close_col='Close', vol_col='Volume'):
    ac = pd.Series([])
    val_last = 0
	
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            val = val_last + ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            val = val_last
        ac.set_value(index, val)
    val_last = val

    ema_long = ac.ewm(ignore_na=False, min_periods=0, com=periods_long, adjust=True).mean()
    ema_short = ac.ewm(ignore_na=False, min_periods=0, com=periods_short, adjust=True).mean()
    data['ChOsc'] = ema_short - ema_long

    return data

def directional_movement_index(data, periods=14, high_col='High', low_col='Low'):
    data = data.reset_index()
    remove_tr_col = False
    if not 'true_range' in data.columns:
        data = average_true_range(data, drop_tr = False)
        remove_tr_col = True

    data['m_plus'] = 0.
    data['m_minus'] = 0.
    
    for i,row in data.iterrows():
        if i>0:
            data.set_value(i, 'm_plus', row[high_col] - data.at[i-1, high_col])
            data.set_value(i, 'm_minus', row[low_col] - data.at[i-1, low_col])
    
    data['dm_plus'] = 0.
    data['dm_minus'] = 0.
    
    for i,row in data.iterrows():
        if row['m_plus'] > row['m_minus'] and row['m_plus'] > 0:
            data.set_value(i, 'dm_plus', row['m_plus'])
            
        if row['m_minus'] > row['m_plus'] and row['m_minus'] > 0:
            data.set_value(i, 'dm_minus', row['m_minus'])
    
    data['di_plus'] = (data['dm_plus'] / data['true_range']).ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data['di_minus'] = (data['dm_minus'] / data['true_range']).ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    
    data['dxi'] = np.abs(data['di_plus'] - data['di_minus']) / (data['di_plus'] + data['di_minus'])
    data.set_value(0, 'dxi',1.)
    data['adx'] = data['dxi'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data = data.drop(['m_plus', 'm_minus', 'dm_plus', 'dm_minus'], axis=1)
    if remove_tr_col:
        data = data.drop(['true_range'], axis=1)
    data = data.set_index('Date')
    return data

def money_flow_index(data, periods=14, vol_col='Volume'):
    data = data.reset_index()
    data['typical_price'] = (data['High'] + data['Low'] + data['Close']) / 3
    
    data['money_flow'] = data['typical_price'] * data[vol_col]
    data['money_ratio'] = 0.
    data['money_flow_index'] = 0.
    data['money_flow_positive'] = 0.
    data['money_flow_negative'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            if row['typical_price'] < data.at[index-1, 'typical_price']:
                data.set_value(index, 'money_flow_positive', row['money_flow'])
            else:
                data.set_value(index, 'money_flow_negative', row['money_flow'])
    
        if index >= periods:
            positive_sum = data['money_flow_positive'][index-periods:index].sum()
            negative_sum = data['money_flow_negative'][index-periods:index].sum()

            if negative_sum == 0.:
				#this is to avoid division by zero below
                negative_sum = 0.00001
            m_r = positive_sum / negative_sum

            mfi = 1-(1 / (1 + m_r))

            data.set_value(index, 'money_ratio', m_r)
            data.set_value(index, 'money_flow_index', mfi)
          
    data = data.drop(['money_flow', 'money_ratio', 'money_flow_positive', 'money_flow_negative'], axis=1)
    
    return data

def williams_ad(data, high_col='High', low_col='Low', close_col='Close'):
    data = data.reset_index()
    data['WillAD'] = 0.
    data.reset_index()
    for index,row in data.iterrows():
        if index > 0:
            prev_value = data.at[index-1, 'WillAD']
            prev_close = data.at[index-1, close_col]
            if row[close_col] > prev_close:
                ad = row[close_col] - min(prev_close, row[low_col])
            elif row[close_col] < prev_close:
                ad = row[close_col] - max(prev_close, row[high_col])
            else:
                ad = 0.
                                                                                                        
            data.set_value(index, 'WillAD', (ad+prev_value))
    data = data.set_index('Date')
    return data

def williams_r(data, periods=14, high_col='High', low_col='Low', close_col='Close'):
    data = data.reset_index()
    data['WillR'] = 0.
    
    for index,row in data.iterrows():
        if index > periods:
            data.set_value(index, 'WillR', ((max(data[high_col][index-periods:index]) - row[close_col]) / 
                                                 (max(data[high_col][index-periods:index]) - min(data[low_col][index-periods:index]))))
    data = data.set_index('Date')
    return data

def trix(data, periods=14, signal_periods=9, close_col='Close'):
    data['TRIX'] = data[close_col].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data['TRIX'] = data['TRIX'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data['TRIX'] = data['TRIX'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data['TRIXSignal'] = data['TRIX'].ewm(ignore_na=False, min_periods=0, com=signal_periods, adjust=True).mean()
        
    return data
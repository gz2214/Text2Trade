import numpy as np
import pandas as pd
import pyarrow
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler

def pre_stock(path):
    '''
    This constructs the daily stock market dataframe: {date, price movement}
    path: path to stock data
    '''
    # retreive stock data
    df = pd.read_csv(path)

    #compute daily market price mean(stock_issued * stock_price)
    df['Date'] = df['Date']
    diff = df['Open'][1:].to_numpy() - df['Open'][:-1].to_numpy() # compute price movement
    price_movement = (diff > 0).astype(int)
    daily = pd.DataFrame({'date': df['Date'][1:], 'price_movement': price_movement})

    # save as csv
    daily.to_csv('../data/daily_price_movement.csv', index=False)

def create_dataset(data, lookback):
    '''
    This function creates a dataset for time series forecasting, with a rolling window of lookback. 
    Note that the first column need to be the daily price movement
    
    Parameters
    data: a 2D numpy array with shape (# of days, # of features)
    lookback: an integer of how many trading days to lookback to
    '''
    assert type(data) == np.ndarray and len(data.shape) == 2, 'Input data needs to be a 2D numpy array'

    n_data, n_feat = data.shape
    loop = n_data - lookback - 1
    X = np.empty((loop, lookback, n_feat))
    y = np.empty((loop, lookback, 1))
    data = data[1:]
    for i in range(loop):
        feature = data[i:i+lookback]
        target = data[i+1:i+lookback+1, 0].reshape(-1,1)
        X[i] = feature
        y[i] = target
    return torch.tensor(X), torch.tensor(y)
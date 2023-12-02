import numpy as np
import pandas as pd
import pyarrow
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
import re

def tensor_string_to_numpy(tensor_str): # tensor saved as csv will become string, so this function converts them into numpy directly
    if pd.isna(tensor_str):
        return np.array([])
    nums = re.findall(r'[\d.\d]+', tensor_str)
    nums = [float(num) for num in nums]
    return np.array([nums])

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
    daily = pd.DataFrame({'Date': df['Date'][1:], 'price_movement': price_movement})

    # save as csv
    daily.to_csv('../data/daily_price_movement.csv', index=False)

def create_dataset(data, lookback, window_size=50, val_step=1, test_step=7, dates=None):
    '''
    This function creates a dataset for time series forecasting, with a rolling window of lookback. 
    Note that the first column need to be the daily price movement
    
    Parameters
    data: a 2D numpy array with shape (# of days, # of features)
    lookback: an integer of how many trading days to lookback to
    window_size: number of days to include in each block, window_size = train_step + test_step
    test_step: number of days to predict
    dates: spcific dates provided for splitting train and val set

    Returns
    X_train, y_train, X_val, y_val: dict of n_blocks as keys and 3D tensor of shape (n_data, lookback, n_feat) as values
    y_train, y_val: dict of n_blocks as keys and 2D tensor of shape (n_data, 1) as values
    X_test: 2D tensor of shape (n_data, n_feat)
    y_test: 2D tensor of shape (n_data, 1)
    '''
    assert type(data) == np.ndarray and len(data.shape) == 2, 'Input data needs to be a 2D numpy array'

    n_data, n_feat = data.shape
    loop = n_data - lookback - 1
    
    # X and y are 3D arrays
    X = np.empty((loop, lookback, n_feat))
    y = np.empty((loop, 1))
    for i in range(loop):
        X[i] = data[i:i+lookback] # all features of the past lookback days
        y[i] = data[i+lookback+1, 0] # price movement of the next day
    
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y)
    #split data into train and val, also keep 7 days at the end as holdout test set
    X_train, X_val, X_test = time_series_split(X, window_size=window_size, val_step=val_step, test_step=test_step)
    y_train, y_val, y_test = time_series_split(y, window_size=window_size, val_step=val_step, test_step=test_step)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
def time_series_split(data, window_size=50, val_step=1, test_step=7):
    """
    Split data using rolling-window (block) split
    data: a 3D array([number of data, number of lookbacks, number of features]) X or a 2D array([number of data, 1]) y
    window_size: number of days to include in each block, window_size = train_step + val_step
    test_step: number of days to predict
    
    return: dict of 3D tensor(train/val, # of lookbacks, # of features) and a 3D tensor(test, # of lookbacks, # of features)
    """
    assert len(data) >= window_size, 'Data length needs to be longer than window size'

    n_data = data.shape[0]
    n_block = n_data // window_size
    test = data[-test_step:] # holdout test set
    data = data[:-test_step] # remove holdout part then split
    train, val= dict(), dict()

    if val_step != 0:
        for i in range(n_block):
            init = i * window_size
            if init + window_size <= n_data:
                block = data[init:init + window_size]
                train[f'block_{i}'] = block[:-val_step]
                val[f'block_{i}'] = block[-val_step:]
            else:
                # Handle the last block which might be smaller
                block = data[init:]
                train[f'block_{i}'] = block[:-val_step] if len(block) > val_step else block
                val[f'block_{i}'] = block[-val_step:] if len(block) > val_step else torch.tensor([])

        return train, val, test
    
    else: # after tuning, we will train the final model on the complete data excluding the holdout test set without splitting into blocks
        return data, torch.tensor([]), test

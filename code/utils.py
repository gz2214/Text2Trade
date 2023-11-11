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
    diff = df['Close'][1:].to_numpy() - df['Close'][:-1].to_numpy() # compute price movement
    price_movement = (diff > 0).astype(int)
    daily = pd.DataFrame({'date': df['Date'][1:], 'price_movement': price_movement})

    # save as csv
    daily.to_csv('../data/daily_price_movement.csv', index=False)
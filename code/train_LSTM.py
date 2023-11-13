import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as data
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from utils import create_dataset
from models import LSTMModel
import copy

def train_model(params, norm_train, norm_val, n_epochs=2000, baseline=False):
    # set hyperparameters
    lookback = params['lookback']
    lr = params['lr']
    n_nodes = params['n_nodes']
    n_layers = params['n_layers']
    dropout_rate = params['dropout_rate'] #usually between 0.2 and 0.5
    
    if baseline:
        # create dataset
        X_train, y_train = create_dataset(norm_train, lookback=lookback)
        X_val, y_val = create_dataset(norm_val, lookback=lookback)
        print(f'X_train shape, y_train shape: {X_train.shape}, {y_train.shape}')
        input_dim = X_train.shape[2]
    
    else:
        pass # fill this in for LSTM with text input 

    # create dataloader
    batch_size = 10
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    # setup model
    model = LSTMModel(input_dim=input_dim, n_nodes=n_nodes, output_dim=1, n_layers=n_layers, dropout_rate=dropout_rate)
    model.double()
    optimizer = opt.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # train model
    min_val_loss = float('inf')
    eval_period = 50 if n_epochs > 1000 else 5

    for t in range(n_epochs):    
        model.train()
        
        for feat, label in loader:
            y_pred = model(feat)
            loss = loss_fn(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        if t % eval_period == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = loss_fn(val_pred, y_val).item()
                # Check if current val loss is lower than the minimum val loss
                if val_loss < min_val_loss:
                    # If so, update the minimum val loss and save the current model state
                    min_val_loss = val_loss
                    opt_model_state = copy.deepcopy(model.state_dict())
                    counter = 0
                else: 
                    counter += 1

        # If the minimum val loss has not been updated for 2 consecutive epochs, stop training
        if counter == 3:
            print(f'Early stopping at epoch {t*eval_period}')
            break
    return min_val_loss, opt_model_state


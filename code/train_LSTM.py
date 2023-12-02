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

# the training process should only take in one block at a time
def train_model(X_train, y_train, model, lr, n_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# create dataset and time series split for two conditions (tuning process and final model training)
    print(f'X_train shape, y_train shape: {X_train.shape}, {y_train.shape}')
    # create dataloader
    batch_size = 10
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=batch_size)
    # setup model
    optimizer = opt.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # train model
    min_loss = float('inf')
    best_model_state = None

    for t in range(n_epochs):
        model.train()
        
        for feat, label in loader:
            feat, label = feat.to(device), label.to(device)
            y_pred = model(feat)
            loss = loss_fn(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update minimum loss and save model state if this is the best model so far
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_model_state = model.state_dict().copy()  # Make a copy of the model state
    print(f'minimum BCElogistic: {min_loss}')

    # Return the best model state and the minimum loss
    return min_loss, best_model_state

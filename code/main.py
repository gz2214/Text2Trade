import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import create_dataset, time_series_split
from models import LSTMModel
from train_LSTM import train_model
from tuning import tune_model
import json
import torch
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, accuracy_score
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(baseline=False):
    # retreive data
    stock_path = '../data/daily_price_movement.csv'
    daily = pd.read_csv(stock_path, header=0) # format: 2D array with (str(date), str()price_movement) 
    
    data = daily.to_numpy()[:, 1].reshape(-1, 1)

    # train and tune model
    print('tuning model...')
    best_params, val_loss = tune_model(data, baseline=baseline)
    print(f'tuning complete.')

    print(f'best hyperparameters: {best_params}\nval loss: {val_loss}')
    if baseline:
        with open('../../results/best_params_baseline.json', 'w') as f:
            json.dump(best_params, f)
    else:
        with open('../../results/best_params_sentiment.json', 'w') as f:
            json.dump(best_params, f)

    # train model with best hyperparameters
    lookback = best_params['lookback']
    lr = best_params['lr']
    n_nodes = best_params['n_nodes']
    n_layers = best_params['n_layers']
    dropout_rate = best_params['dropout_rate']

    X_train, _, X_test, y_train, _, y_test = create_dataset(data, lookback=lookback, window_size=50, val_step=0, test_step=7)
    min_test_loss, opt_model_state = train_model(best_params, X_train, y_train, n_epochs=1000)
    print(f'minimum test BCElogistic: {min_test_loss}')

    # setup model with the optimized weights and hyperparams
    model = LSTMModel(input_dim=data.shape[1], n_nodes=n_nodes, output_dim=1, n_layers=n_layers, dropout_rate=dropout_rate)
    #model.double()
    opt_model_state_cpu = {key: value.cpu() for key, value in opt_model_state.items()}
    model.load_state_dict(opt_model_state_cpu) # load the optimal model state
    model.eval()

    # evaluate model with test set
    with torch.no_grad():
        output = model(X_test)

    t_pred = output[:, -1, 0].view(-1).numpy()  # select the 1-D output and reshape from (batch_size, sequence_length) to (batch_size*sequence_length, 1)

    # Calculate true positive and false positive rates
    y_true = y_test[:,0].view(-1).numpy()
    y_score = t_pred
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if baseline:
        plt.savefig('../../results/roc_baseline.png')
    else:
        plt.savefig('../../results/roc_sentiment.png')
    plt.show()

    # Calculate precision, f1, and accuracy score
    precision = precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_score)

    results_str = f"Precision: {precision}\nF1 Score: {f1}\nAccuracy: {accuracy}"
    if baseline:
        with open('../../results/eval_baseline.txt', 'w') as f:
            f.write(results_str)
    else:
        with open('../../results/eval_sentiment.txt', 'w') as f:
            f.write(results_str)

    # Print the scores
    print(f'Precision: {precision}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    if sys.argv[1].lower() not in ['true', 'false']:
        raise ValueError('Invalid input: please use True or False')
    baseline = sys.argv[1].lower() == 'true'
    main(baseline=baseline)
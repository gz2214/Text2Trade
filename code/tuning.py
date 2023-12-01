import optuna
import optuna.visualization as vis
from train_LSTM import train_model
from utils import create_dataset
from models import LSTMModel
import torch
import torch.nn as nn
import numpy as np

def objective(trial, data):
    # Define hyperparameters
    lookback = trial.suggest_int("lookback", 5, 60, step=5)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_nodes = trial.suggest_int("n_nodes", 10, 100, step=10)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)

    # Create dataset with the current trial's lookback value
    X_train, X_val, _, y_train, y_val, _ = create_dataset(data, lookback=lookback, window_size=50, val_step=1, test_step=7)

    # train then get validation loss
    model = LSTMModel(input_dim=data.shape[1], n_nodes=n_nodes, output_dim=1, n_layers=n_layers, dropout_rate=dropout_rate)
    #model.double()
    blocks = list(X_train.keys())
    num_blocks = len(blocks)
    val_loss_list = np.zeros(num_blocks) # list of validation loss for each block

    for i, block in enumerate(blocks): # train block by block
        _, best_set = train_model(X_train[block], y_train[block], model, lr=lr, n_epochs=500)
        # evaluate with val set of single block
        best_set = {key: value.cpu() for key, value in best_set.items()}
        model.load_state_dict(best_set)
        model.eval()
        with torch.no_grad():
            output = model(X_val[block])
        
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(output, y_val[block])
        val_loss_list[i] = (loss.item())

    return val_loss_list.mean()

def study_early_stop(study, trial):
    # Stop if no improvement in the last N trials
    N = 30
    threshold = 0.001

    if len(study.trials) < N:
        return

    values = [t.value for t in study.trials[-N:]]
    best_value = min(values)
    if all((abs(v - best_value) < threshold) or (v > best_value) for v in values):
        study.stop()

def tune_model(data, baseline=False, best_params=None):
    study = optuna.create_study(direction="minimize")
    if best_params: # If this is not the first block, then initialize BayesOpt with the best_params from the previous block
        study.enqueue_trial({
            'lookback': best_params['lookback'],
            'lr': best_params['lr'],
            'n_nodes': best_params['n_nodes'],
            'n_layers': best_params['n_layers'],
            'dropout_rate': best_params['dropout_rate']
            })
    study.optimize(lambda trial: objective(trial, data), n_trials=200, callbacks=[study_early_stop])

    best_trial = study.best_trial
    best_params = best_trial.params
    best_val_loss = best_trial.value

    # Plot optimization history
    history = vis.plot_optimization_history(study)
    history.show()
    if baseline:
        history.write_image('../results/optimization_history_baseline.png')
    else:
        history.write_image('../results/optimization_history_sentiment.png')

    # Plot parameter relationship
    importance = vis.plot_param_importances(study)
    importance.show()
    if baseline:
        importance.write_image('../results/param_importance_baseline.png')
    else:
        importance.write_image('../results/param_importance_sentiment.png')
    
    # Plot slice of the parameters
    slice = vis.plot_slice(study, params=['n_layers', 'n_nodes', 'dropout_rate', 'lr'])
    slice.show()
    if baseline:
        slice.write_image('../results/param_slice_baseline.png')
    else:
        slice.write_image('../results/param_slice_sentiment.png')


    return best_params, best_val_loss
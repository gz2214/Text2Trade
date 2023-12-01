import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, n_nodes, output_dim=1, n_layers=1, dropout_rate=0.0):
        # input is in format (batch_size, seq_len, num_features)
        super().__init__()
        self.input_dim = input_dim
        self.n_nodes = n_nodes
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # Define LSTM layer(s)
        self.lstm = nn.LSTM(self.input_dim, self.n_nodes, self.n_layers, bias=True, batch_first=True, dropout=dropout_rate if n_layers>1 else 0)
        
        # Define the output layer
        self.linear = nn.Linear(self.n_nodes, output_dim)
        
    def forward(self, x):
        # output is in format (batch_size, seq_len, output_dim)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]

        # 'last_out' is (batch_size, hidden_size)
        pred = self.linear(last_out)
        
        return pred
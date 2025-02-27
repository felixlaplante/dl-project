#####################
# Configuration file

import torch

#####################
# Hyperparameters

ff_freq_doy = 12
ff_freq_mod = 12

weather_cols = ['t', 'tminsol', 'ff', 'pmer']

seq_length = 48

lr = 1e-3
batch_size = 128
epochs = 10
hidden_size = 256
num_layers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
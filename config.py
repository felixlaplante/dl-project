#####################
# Configuration file

import torch

#####################
# Hyperparameters

ff_freq_doy = 12
ff_freq_mod = 12

weather_cols = ['t', 'tminsol', 'ff', 'pmer']

seq_length = 48

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
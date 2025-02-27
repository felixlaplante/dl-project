#####################
# Set working directory

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#####################
# Import libraries

import joblib
import zipfile

import numpy as np
import pandas as pd

import torch

#####################
# Import custom functions

import config
from prep import load_ff_df, load_date_df, load_weather_df, load_holiday_df, load_fr_co_emissions_df, load_seq, get_scalers
from fun import LSTM, predict

#####################
# Set seed

np.random.seed(42)
torch.manual_seed(42)

#####################
# Load data

train_df = pd.read_csv('data/train.csv')
train_df['date'] = pd.to_datetime(train_df['date'], utc=True)
train_df.set_index('date', inplace=True)

test_df = pd.read_csv('data/test.csv')
test_df['date'] = pd.to_datetime(test_df['date'], utc=True)
test_df.set_index('date', inplace=True)

weather_df = pd.read_parquet('data/meteo.parquet')
weather_df['date'] = pd.to_datetime(weather_df['date'], utc=True)
weather_df.set_index('date', inplace=True)

co_emissions_df = pd.read_csv('data/annual-co-emissions.csv')
co_emissions_df['Year'] = pd.to_datetime(co_emissions_df['Year'].astype(str) + '-07-01', utc=True)
co_emissions_df.set_index('Year', inplace=True)

#####################
# Prepare data

warmump_test_df = pd.concat([train_df[-config.seq_length+1:], test_df], axis=0)

X_warmup_test_ff_df = load_ff_df(warmump_test_df)
X_warmup_test_date_df = load_date_df(warmump_test_df)
X_warmup_test_weather_df = load_weather_df(weather_df, warmump_test_df).interpolate(method='time', limit_direction='both')
X_warmup_test_holiday_df = load_holiday_df(warmump_test_df)
X_warmup_test_fr_co_emissions_df = load_fr_co_emissions_df(co_emissions_df, warmump_test_df)

X_warmump_test_df = pd.concat([X_warmup_test_ff_df, X_warmup_test_date_df, X_warmup_test_weather_df, X_warmup_test_holiday_df, X_warmup_test_fr_co_emissions_df], axis=1)
X_warmup_test = X_warmump_test_df.values.astype('float32')

####################
# Scale data

X_scaler = joblib.load('scalers/X_scaler.pkl')
y_scaler = joblib.load('scalers/y_scaler.pkl')

X_warmup_test_scaled = X_scaler.transform(X_warmup_test)

####################
# Load model

input_dim = X_warmup_test.shape[1]
output_dim = len(y_scaler.scale_)

model = LSTM(input_dim, output_dim, config.hidden_size, config.num_layers).to(config.device)
model.load_state_dict(torch.load('model/model.pth', weights_only=True))

####################
# Predict

y_pred_scaled = predict(model, X_warmup_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

#####################
# Save

predictions_df = pd.DataFrame(y_pred, index=test_df.index, columns=train_df.columns)
predictions_df = predictions_df.add_prefix('pred_')
predictions_df.index = predictions_df.index.tz_convert('Europe/Paris')

predictions_df.to_csv('pred/pred.csv', index=True)

with zipfile.ZipFile('pred/submission.zip', 'w') as zipf:
    zipf.write('pred/pred.csv', arcname='pred.csv')

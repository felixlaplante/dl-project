#####################
# Set working directory

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#####################
# Import libraries

import joblib
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

#####################
# Import custom functions

import config
from prep import load_ff_df, load_date_df, load_weather_df, load_holiday_df, load_fr_co_emissions_df, load_seq, get_scalers
from fun import RMSELoss, train_loop, Load_Xy, LSTM

#####################
# Set seed

np.random.seed(42)
torch.manual_seed(42)

#####################
# Load data

train_df = pd.read_csv('data/train.csv')
train_df['date'] = pd.to_datetime(train_df['date'], utc=True)
train_df.set_index('date', inplace=True)

weather_df = pd.read_parquet('data/meteo.parquet')
weather_df['date'] = pd.to_datetime(weather_df['date'], utc=True)
weather_df.set_index('date', inplace=True)

co_emissions_df = pd.read_csv('data/annual-co-emissions.csv')
co_emissions_df['Year'] = pd.to_datetime(co_emissions_df['Year'].astype(str) + '-07-01', utc=True)
co_emissions_df.set_index('Year', inplace=True)

#####################
# Prepare data

X_ff_df = load_ff_df(train_df)
X_date_df = load_date_df(train_df)
X_weather_df = load_weather_df(weather_df, train_df)
X_holiday_df = load_holiday_df(train_df)
X_fr_co_emissions_df = load_fr_co_emissions_df(co_emissions_df, train_df)

X_df = pd.concat([X_ff_df, X_date_df, X_weather_df, X_holiday_df, X_fr_co_emissions_df], axis=1)
y_df = train_df

X = X_df.values.astype('float32')
y = y_df.values.astype('float32')

X_scaler, y_scaler = get_scalers(X, y)

train_X, train_y = X_scaler.transform(X), y_scaler.transform(y)

train_full_dataset = Load_Xy(*load_seq(train_X, train_y))
train_full_loader = DataLoader(train_full_dataset, batch_size=config.batch_size, shuffle=True)

#####################
# Train model

loss_fn = RMSELoss(torch.FloatTensor(y_scaler.scale_).to(config.device))

input_dim = X.shape[1]
output_dim = y.shape[1]

model = LSTM(input_dim, output_dim, config.hidden_size, config.num_layers).to(config.device)

optimizer = optim.AdamW(model.parameters(), lr=config.lr)

train_losses = []
for epoch in tqdm(range(config.epochs), desc="Training"):
    t_loss = train_loop(train_full_loader, model, loss_fn, optimizer)
    train_losses.append(t_loss)

    if (epoch + 1) % 2 == 0:
        print(f" Epoch {epoch+1}: Train Loss: {t_loss:.4f}")

plt.plot(np.array(train_losses), label='Train Loss')
plt.legend()
plt.show()

#####################
# Save model

joblib.dump(X_scaler, 'scalers/X_scaler.pkl')
joblib.dump(y_scaler, 'scalers/y_scaler.pkl')

torch.save(model.state_dict(), 'model/model.pth')

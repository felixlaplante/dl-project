#####################
# Set working directory

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#####################
# Import libraries

import scipy.stats as stats
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

#####################
# Import custom functions

import config
from prep import load_ff_df, load_date_df, load_weather_df, load_holiday_df, load_fr_co_emissions_df, load_seq, get_scalers
from fun import RMSELoss, validate, predict, train_loop, Load_Xy, LSTM

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

train_X_df, val_X_df = train_test_split(X_df, test_size=0.15, shuffle=False)
train_y_df, val_y_df = train_test_split(y_df, test_size=0.15, shuffle=False)

train_X = train_X_df.values.astype('float32')
train_y = train_y_df.values.astype('float32')
val_X = val_X_df.values.astype('float32')
val_y = val_y_df.values.astype('float32')

X_scaler, y_scaler = get_scalers(train_X, train_y)

train_X, train_y = X_scaler.transform(train_X), y_scaler.transform(train_y)
val_X, val_y = X_scaler.transform(val_X), y_scaler.transform(val_y)

train_dataset = Load_Xy(*load_seq(train_X, train_y))
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

val_dataset = Load_Xy(*load_seq(val_X, val_y))
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

#####################
# Train model

loss_fn = RMSELoss(torch.FloatTensor(y_scaler.scale_).to(config.device))

input_dim = train_X.shape[1]
output_dim = train_y.shape[1]

model = LSTM(input_dim, output_dim, config.hidden_size, config.num_layers).to(config.device)

optimizer = optim.AdamW(model.parameters(), lr=config.lr)

train_losses, val_losses = [], []
for epoch in tqdm(range(config.epochs), desc="Training"):
    t_loss = train_loop(train_loader, model, loss_fn, optimizer)
    train_losses.append(t_loss)
    v_loss = validate(val_loader, model, loss_fn)
    val_losses.append(v_loss)

    if (epoch + 1) % 2 == 0:
        print(f" Epoch {epoch+1}: Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")

train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

#####################
# Plot losses

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1, config.epochs+1), y=train_losses, mode='lines', name='Train Loss'))
fig.add_trace(go.Scatter(x=np.arange(1, config.epochs+1), y=val_losses, mode='lines', name='Val Loss'))

fig.update_layout(title='Train and Validation Losses',
                  xaxis_title='Epoch',
                  yaxis_title='Loss',
                  ) 

fig.write_image('val_visualization/losses.png', scale=2)

#####################
# Calculate loss on validation set

X_warmup_val_scaled = np.vstack([train_X[-config.seq_length+1:], val_X])
y_pred_scaled = predict(model, X_warmup_val_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(val_y)

predictions_df = pd.DataFrame(y_pred, index=val_y_df.index, columns=val_y_df.columns)
predictions_df = predictions_df.add_prefix('pred_')

mape = mean_absolute_percentage_error(val_y_df['France'], predictions_df['pred_France'])
print(f'France MAPE: {mape:.4f}')

rmse = np.sqrt(np.nanmean((val_y_df.values - predictions_df.values)**2, axis=0)).sum()
print(f'Total RMSE: {rmse:.4f}')

#####################
# Plot predictions yearly

fig = go.Figure()
fig.add_trace(go.Scatter(x=val_y_df.index, y=val_y_df['France'].values, mode='lines', name='France load'))
fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df['pred_France'], mode='lines', name='Predicted France load'))
fig.add_trace(go.Scatter(x=val_y_df.index, y=val_y_df['France'].rolling(window=48*30, center=True).mean().values.ravel(), mode='lines', name='Smoothed France load'))
fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df['pred_France'].rolling(window=48*30, center=True).mean().values.ravel(), mode='lines', name='Smoothed Predicted France load'))

fig.update_layout(title='Predictions on the validation set',
                  xaxis_title='Date',
                  yaxis_title='Load',
                  )
fig.write_image('val_visualization/predictions_yr.png', scale=2)

#####################
# Plot predictions weekly

fig = go.Figure()
fig.add_trace(go.Scatter(x=val_y_df.index[:48*7], y=val_y_df['France'].values[:48*7], mode='lines', name='France load'))
fig.add_trace(go.Scatter(x=predictions_df.index[:48*7], y=predictions_df['pred_France'][:48*7], mode='lines', name='Predicted France load'))

fig.update_layout(title='Predictions on the first week of the validation set',
                  xaxis_title='Date',
                  yaxis_title='Load',
                  )
fig.write_image('val_visualization/predictions_wk.png', scale=2)

#####################
# Histogram of errors

errors = val_y_df['France'].values - predictions_df['pred_France'].values
fig = go.Figure()
fig.add_trace(go.Histogram(x=errors, nbinsx=100, histnorm='percent', name='Errors'))
fig.update_layout(title='Histogram of errors',
                  xaxis_title='Error',
                  yaxis_title='Percentage',
                  )
fig.write_image('val_visualization/errors.png', scale=2)

#####################
# q-q plot of standarduzed errors

standardized_errors = (errors - np.mean(errors)) / np.std(errors)
    
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
sample_quantiles = np.quantile(standardized_errors, np.linspace(0.01, 0.99, 100)) 

fig = go.Figure()
fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name='Quantiles'))
min_val = min(min(theoretical_quantiles), min(sample_quantiles))
max_val = max(max(theoretical_quantiles), max(sample_quantiles))
fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(dash='dash'), name='y=x'))

fig.update_layout(
    title='QQ Plot of standardized residuals',
    xaxis_title='Theoretical quantiles',
    yaxis_title='Sample quantiles',
)
fig.write_image('val_visualization/qq_plot.png', scale=2)
import holidays

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import config


def load_ff_df(df):
    def f_features(x, T, freq):
        freqs = np.arange(1, freq + 1).reshape(1, -1)
        f_cos = np.cos(2 * np.pi * freqs * x / T)
        f_sin = np.sin(2 * np.pi * freqs * x / T)
        return np.hstack([f_cos, f_sin])

    doy = df.index.day_of_year.values.reshape(-1, 1)
    mod = df.index.astype('int64').values.reshape(-1, 1) // 1e9 % 86400

    ff_doy = f_features(doy, 365, config.ff_freq_doy)
    ff_mod = f_features(mod, 86400, config.ff_freq_mod)

    ff_df = pd.DataFrame(data=np.hstack([ff_doy, ff_mod]), index=df.index)

    return ff_df


def load_date_df(df):
    time_df = pd.DataFrame(data=df.index.astype('int64'))
    weekday_encoded_df = pd.get_dummies(df.index.weekday, prefix='weekday')
    dls_series = pd.Series(df.index).dt.tz_convert('Europe/Paris').apply(lambda x: int(x.dst() != pd.Timedelta(0)))
    dls_encoded_df = pd.get_dummies(dls_series, prefix='dls').astype(int)
    date_df = pd.concat([time_df, weekday_encoded_df, dls_encoded_df], axis=1)
    date_df.index = df.index 

    return date_df


def load_holiday_df(df):
    fr_holidays = holidays.FR(years=range(df.index.min().year, df.index.max().year + 1))
    holiday_series = pd.Series({date: fr_holidays.get(date, 'No Holiday') for date in df.index}, index=df.index, name='holiday')
    holiday_df = pd.get_dummies(holiday_series, prefix='holiday')
    holiday_df.index = df.index 
    
    return holiday_df


def load_weather_df(wdf, df):
    weather_df = wdf.copy()
    
    for col in config.weather_cols:
        weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
    
    weather_df = weather_df.pivot_table(
        index='date',
        columns='numer_sta',
        values=config.weather_cols
    )
    
    weather_df.dropna(thresh=int(0.95 * len(weather_df)), axis=1, inplace=True)
    weather_df = weather_df.resample('30min').interpolate(method='cubic', limit_direction='both')
    weather_df = weather_df.reindex(df.index)
    
    return weather_df


def load_fr_co_emissions_df(codf, df):
    fr_co_emissions_df = codf[codf['Entity'] == 'France'].copy()
    fr_co_emissions_df = fr_co_emissions_df.drop(columns='Entity')
    fr_co_emissions_df = fr_co_emissions_df.resample('30min').interpolate(method='cubic', limit_direction='both')
    fr_co_emissions_df = fr_co_emissions_df.reindex(df.index)
    
    return fr_co_emissions_df


def get_scalers(X, y):
    continuous_cols_idxs = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > 2]
    X_scaler= ColumnTransformer([('scaler', StandardScaler(), continuous_cols_idxs)], remainder='passthrough').fit(X)
    y_scaler = StandardScaler().fit(y)
    return X_scaler, y_scaler


def load_seq(X, y):
    X_seq = []
    y_seq = None if y is None else y[config.seq_length-1:]
    
    for i in range(X.shape[0] - config.seq_length + 1):
        seq = X[i:i+config.seq_length]
        X_seq.append(seq)

    return np.array(X_seq), y_seq  
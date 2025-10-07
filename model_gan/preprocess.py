# model_gan/preprocess.py
import numpy as np
import pandas as pd

def load_and_preprocess():
    file_path = "../data/LD2011_2014.txt"
    try:
        data = pd.read_csv(
            file_path,
            sep=';',
            index_col=0,
            parse_dates=True,
            low_memory=False,
            decimal=','  # dataset uses comma decimals
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Put LD2011_2014.txt at {file_path}")

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
    data = data.sort_index()

    # ensure numeric
    for c in data.columns:
        if not np.issubdtype(data[c].dtype, np.number):
            data[c] = pd.to_numeric(data[c].astype(str).str.replace(',', '.'), errors='coerce')
    data = data.fillna(0.0)

    # resample to hourly
    data_hourly = data.resample('1h').mean()

    # 24-d average-day profile per client
    feature_list = []
    for client in range(data_hourly.shape[1]):
        series = data_hourly.iloc[:, client]
        hourly_means = series.groupby(series.index.hour).mean().reindex(range(24), fill_value=0.0)
        feature_list.append(hourly_means.values)

    X = np.array(feature_list, dtype=np.float32)  # (370, 24)

    # min–max per hour
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    rng = np.where((X_max - X_min) == 0, 1, (X_max - X_min))
    X = (X - X_min) / rng

    return X.astype(np.float32)

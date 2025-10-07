# model_gan/preprocess.py
import numpy as np
import pandas as pd

def load_and_preprocess():
    file_path = "../data/LD2011_2014.txt"
    try:
        # Parse datetime index, handle decimal comma, and avoid mixed-type chunks
        data = pd.read_csv(
            file_path,
            sep=';',
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            low_memory=False,
            decimal=','  # <-- key for this dataset
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset file not found at {file_path}. "
            "Please place LD2011_2014.txt in the data/ folder."
        )

    # If for any reason the index still isn't datetime, coerce it
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
    data = data.sort_index()

    # Ensure ALL value columns are numeric (in case any slipped in as object)
    for c in data.columns:
        if not np.issubdtype(data[c].dtype, np.number):
            # replace possible commas and coerce
            data[c] = pd.to_numeric(data[c].astype(str).str.replace(',', '.'), errors='coerce')

    # (This dataset is supposed to have no missing values, but coercion may create some NaNs)
    data = data.fillna(0.0)

    # Resample to hourly (non-deprecated alias)
    data_hourly = data.resample('1h').mean()

    # Build a 24-dim “average day” profile per client
    num_clients = data_hourly.shape[1]
    feature_list = []
    for client in range(num_clients):
        series = data_hourly.iloc[:, client]
        hourly_means = series.groupby(series.index.hour).mean().reindex(range(24), fill_value=0.0)
        feature_list.append(hourly_means.values)

    X = np.array(feature_list, dtype=np.float32)  # shape (N_clients, 24)

    # Min–max scale per feature (hour)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = np.where((X_max - X_min) == 0, 1, (X_max - X_min))
    X_norm = (X - X_min) / X_range

    return X_norm.astype(np.float32)

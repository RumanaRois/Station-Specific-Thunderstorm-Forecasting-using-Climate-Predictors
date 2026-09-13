########STL+ML+TS##########
#################################################################Deep Learning###############################################
#####STL+ANN+ETS####
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tbats import TBATS
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_excel('S1_data.xlsx')
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
df = df.set_index('DateTime')
df = df.asfreq('3h')

# STL decomposition with daily seasonality (period=8)
stl = STL(df['DI'], period=8)
res = stl.fit()

df['seasonal'] = res.seasonal
df['trend'] = res.trend
df['DI_sa'] = df['DI'] - df['seasonal']  # seasonally adjusted series
df.dropna(inplace=True)
df.reset_index(inplace=True)

# Extract datetime features
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month

# One-hot encode Season column from original dataset (to align)
original_df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data.xlsx')
original_df['DateTime'] = pd.to_datetime(original_df['DateTime'], errors='coerce')

df['Season'] = original_df.set_index('DateTime').loc[df['DateTime'], 'Season'].values
df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Lag features on seasonally adjusted DI
df['DI_sa_lag_1'] = df['DI_sa'].shift(1)
df['DI_sa_lag_3'] = df['DI_sa'].shift(3)
df['DI_sa_lag_6'] = df['DI_sa'].shift(6)

# Rolling window features on seasonally adjusted DI (7 periods window)
df['rollmean_7'] = df['DI_sa'].rolling(window=7).mean()
df['rollstd_7'] = df['DI_sa'].rolling(window=7).std()

df.dropna(inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

feature_cols = ['DI_sa', 'Rainfall', 'Cloud', 'Atmosphere', 'Wind', 'Hour', 'Day', 'Month',
                'DI_sa_lag_1', 'DI_sa_lag_3', 'DI_sa_lag_6',
                'rollmean_7', 'rollstd_7',
                'Season_Spring', 'Season_Summer', 'Season_Winter']

X = df[feature_cols]
Y = df['DI_sa'].shift(-1).dropna()  # Next step prediction

# align X and Y
X = X.iloc[:-1, :]

# Train-test split (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.2, shuffle=False)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save seasonal component of test set for adding back predictions
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(256, input_shape=(X_train_scaled.shape[1],)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dense(1))

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=2)

# Predict and add back seasonality
y_pred_sa = model.predict(X_test_scaled).flatten()
y_pred = y_pred_sa + seasonal_test

# residuals calculate
residuals_train = y_train - model.predict(X_train_scaled).flatten()

# Assuming the data starts from '2014-01-01' and the frequency is 3 hours ('3h')
start_date = '2014-01-01'
datetime_index = pd.date_range(start=start_date, periods=len(residuals_train), freq='3h')

# Now, assign this index to your residuals series
residuals_series = pd.Series(residuals_train, index=datetime_index)

ets_model = ExponentialSmoothing(residuals_series, trend='add', seasonal='add', seasonal_periods=8).fit()
residuals_forecast = ets_model.forecast(len(y_test))

hybrid_forecast = y_pred_sa + seasonal_test + residuals_forecast.values

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
r2 = r2_score(y_test, hybrid_forecast)

print("STL + ANN + ETS")
print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}%, MASE: {mase:.6f}, R2: {r2:.6f}")
######STL+ANN+SARIMAX#####
sarimax_model = SARIMAX(residuals_series, order=(0,0,1), seasonal_order=(0,0,0,0))
sarimax_fit = sarimax_model.fit(disp=False)
residuals_forecast = sarimax_fit.get_forecast(steps=len(y_test)).predicted_mean

hybrid_forecast = y_pred_sa + seasonal_test + residuals_forecast.values

mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
r2 = r2_score(y_test, hybrid_forecast)

print("STL + ANN + SARIMAX")
print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}%, MASE: {mase:.6f}, R2: {r2:.6f}")
##########STL+ANN+TBATS###########
residuals_series_downsampled = residuals_series.asfreq('6h').dropna()

tbats_model = TBATS(
    seasonal_periods=[4],  # 6h freq means 4 periods per day
    use_arma_errors=False,
    use_box_cox=False,
    use_damped_trend=False,
    n_jobs=1
)

tbats_fit = tbats_model.fit(residuals_series_downsampled)

steps = len(y_test)  # Adjust steps accordingly if you downsampled y_test too
residuals_forecast = tbats_fit.forecast(steps=steps)
hybrid_forecast = y_pred_sa + seasonal_test + residuals_forecast

mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
r2 = r2_score(y_test, hybrid_forecast)

print("STL + ANN + TBATS")
print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}%, MASE: {mase:.6f}, R2: {r2:.6f}")

##########STL+GRU+ETS###########
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load data and STL decomposition
df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data_LSTM.xlsx')
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
df = df.set_index('DateTime').asfreq('3h')

stl = STL(df['DI'], period=8)
res = stl.fit()
df['seasonal'] = res.seasonal
df['trend'] = res.trend
df['DI_sa'] = df['DI'] - df['seasonal']
df.dropna(inplace=True)
df.reset_index(inplace=True)

# Extract date/time features
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month

# One-hot encode Season column aligned from original data
original_df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data_LSTM.xlsx')
original_df['DateTime'] = pd.to_datetime(original_df['DateTime'], errors='coerce')
df['Season'] = original_df.set_index('DateTime').loc[df['DateTime'], 'Season'].values
df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Create lag and rolling features on seasonally adjusted DI
df['DI_sa_lag_1'] = df['DI_sa'].shift(1)
df['DI_sa_lag_3'] = df['DI_sa'].shift(3)
df['DI_sa_lag_6'] = df['DI_sa'].shift(6)
df['rollmean_7'] = df['DI_sa'].rolling(window=7).mean()
df['rollstd_7'] = df['DI_sa'].rolling(window=7).std()
df.dropna(inplace=True)


# Define features and target (next-step prediction)
feature_cols = ['DI_sa', 'Rainfall', 'Cloud', 'Atmosphere', 'Wind', 'Hour', 'Day', 'Month',
                'DI_sa_lag_1', 'DI_sa_lag_3', 'DI_sa_lag_6',
                'rollmean_7', 'rollstd_7',
                'Season_Spring', 'Season_Summer', 'Season_Winter']

X = df[feature_cols]
Y = df['DI_sa'].shift(-1).dropna()
X = X.iloc[:-1, :]
seasonal_test = df['seasonal'].iloc[-len(Y):].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.2, shuffle=False)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create sequences for LSTM
time_steps = 6
features_num = X_train.shape[1]

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)
seasonal_test_seq = seasonal_test[time_steps:]

datetime_test_seq = df['DateTime'].iloc[-len(y_test_seq):].reset_index(drop=True)
seasonal_series = pd.Series(seasonal_test, index=pd.date_range(start=datetime_test_seq.min(), periods=len(seasonal_test), freq='3h'))
seasonal_test_seq_aligned = seasonal_series.loc[datetime_test_seq].values


from tensorflow.keras.layers import GRU

model_gru = Sequential()
model_gru.add(Input(shape=(time_steps, features_num)))
model_gru.add(GRU(units=50, return_sequences=True))
model_gru.add(Dropout(0.3))
model_gru.add(GRU(units=50))
model_gru.add(Dropout(0.3))
model_gru.add(Dense(1))

model_gru.compile(optimizer='adam', loss='mean_squared_error')
model_gru.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# Predict and add back seasonality
y_pred_sa = model_gru.predict(X_test_seq).flatten()
y_pred = y_pred_sa + seasonal_test_seq_aligned

# Residuals calculation on training set
train_pred = model_gru.predict(X_train_seq).flatten()
residuals_train = y_train_seq - train_pred

# ETS model to residuals using exact train datetime index

datetime_index_train = df['DateTime'].iloc[:len(y_train_seq)].reset_index(drop=True)
residuals_series = pd.Series(residuals_train, index=pd.to_datetime(datetime_index_train)).asfreq('3h').dropna()

ets_model = ExponentialSmoothing(residuals_series, trend='add', seasonal='add', seasonal_periods=8).fit()

# Forecast residuals for test length
residuals_forecast = ets_model.forecast(len(y_test_seq))

# Get date range for test sequences (after time_steps offset)
#datetime_test_seq = df['DateTime'].iloc[-len(y_test_seq):].reset_index(drop=True)

# Slice seasonal_test using these dates
#seasonal_test_seq_aligned = df.set_index('DateTime').loc[datetime_test_seq, 'seasonal'].values

#import pandas as pd


hybrid_forecast = y_pred + residuals_forecast.values 

# Metrics calculation
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n -1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

mse = mean_squared_error(y_test_seq, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_seq, hybrid_forecast)
mape = np.mean(np.abs((y_test_seq - hybrid_forecast) / y_test_seq)) * 100
mase = mean_absolute_scaled_error(y_test_seq, hybrid_forecast, y_train_seq)
r2 = r2_score(y_test_seq, hybrid_forecast)

print("STL + GRU + ETS Hybrid Model Metrics:")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"MAPE: {mape:.6f}%")
print(f"MASE: {mase:.6f}")
print(f"R²: {r2:.6f}")
##########STL+GRU+SARIMAX########
# Fit SARIMAX on residuals
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarimax_model = SARIMAX(residuals_series, order=(0,0,1), seasonal_order=(0,0,0,0))
sarimax_fit = sarimax_model.fit(disp=False)

# Forecast residuals for test sequences length
residuals_forecast = sarimax_fit.get_forecast(steps=len(y_test_seq)).predicted_mean

# Predict GRU on test sequences
y_pred_sa = model_gru.predict(X_test_seq).flatten()

# Use aligned seasonal test sequence from your setup
seasonal_test_seq_aligned = seasonal_test_seq_aligned  # provided in your setup

# Compose hybrid forecast
hybrid_forecast = y_pred_sa + seasonal_test_seq_aligned + residuals_forecast.values

# Metrics calculation
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n-1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

mase = mean_absolute_scaled_error(y_test_seq, hybrid_forecast, y_train_seq)
mse = mean_squared_error(y_test_seq, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_seq, hybrid_forecast)
mape = np.mean(np.abs((y_test_seq - hybrid_forecast) / y_test_seq)) * 100
r2 = r2_score(y_test_seq, hybrid_forecast)

print("STL + GRU + SARIMAX Hybrid Model Metrics:")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"MAPE: {mape:.6f}%")
print(f"MASE: {mase:.6f}")
print(f"R²: {r2:.6f}")
########STL+GRU+TBATS##########
# Fit TBATS model on downsampled residuals
from tbats import TBATS

tbats_model = TBATS(seasonal_periods=[8], use_arma_errors=False, use_box_cox=False, use_damped_trend=False, n_jobs=1)
tbats_fit = tbats_model.fit(residuals_series)

# Forecast residuals for test sequences length
steps = len(y_test_seq)
residuals_forecast = tbats_fit.forecast(steps=steps)

# Predict on test sequences
y_pred_sa = model_gru.predict(X_test_seq).flatten()

# Use seasonal test slice aligned to test sequences
seasonal_test_seq_aligned = seasonal_test_seq_aligned  # provided in your setup

# Compose hybrid forecast
hybrid_forecast = y_pred_sa + residuals_forecast + seasonal_test_seq_aligned

# Define MASE metric
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

mase = mean_absolute_scaled_error(y_test_seq, hybrid_forecast, y_train_seq)
mse = mean_squared_error(y_test_seq, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_seq, hybrid_forecast)
mape = np.mean(np.abs((y_test_seq - hybrid_forecast) / y_test_seq)) * 100
r2 = r2_score(y_test_seq, hybrid_forecast)

print("STL + GRU + TBATS Hybrid Model Metrics:")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"MAPE: {mape:.6f}%")
print(f"MASE: {mase:.6f}")
print(f"R²: {r2:.6f}")
############STL+DT+ETS##########
import pandas as pd
from statsmodels.tsa.seasonal import STL

# Load data
df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data.xlsx')
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
df = df.set_index('DateTime')
df = df.asfreq('3h')  # Assuming 3-hour frequency

# STL decomposition with daily seasonality (period=8)
stl = STL(df['DI'], period=8)
res = stl.fit()

df['seasonal'] = res.seasonal
df['trend'] = res.trend
df['DI_sa'] = df['DI'] - df['seasonal']  # Seasonally adjusted series
df.dropna(inplace=True)  # Remove NaNs created by STL edges

# Reset index for feature engineering
df.reset_index(inplace=True)

# Extract datetime features
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month

# One-hot encode Season column from original data
original_df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data.xlsx')
original_df['DateTime'] = pd.to_datetime(original_df['DateTime'], errors='coerce')
# Align and add Season column
df['Season'] = original_df.set_index('DateTime').loc[df['DateTime'], 'Season'].values
df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Lag features on seasonally adjusted DI
df['DI_sa_lag_1'] = df['DI_sa'].shift(1)
df['DI_sa_lag_3'] = df['DI_sa'].shift(3)
df['DI_sa_lag_6'] = df['DI_sa'].shift(6)

df.dropna(inplace=True)

from sklearn.model_selection import train_test_split

features = ['Rainfall', 'Cloud', 'Atmosphere', 'Wind', 'Hour', 'Day', 'Month',
            'DI_sa_lag_1', 'DI_sa_lag_3', 'DI_sa_lag_6',
            'Season_Spring', 'Season_Summer', 'Season_Winter']

X = df[features]
y = df['DI_sa']  # seasonally adjusted target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#dt_model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=100, random_state=42)
dt_model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=100, random_state=42)
dt_model.fit(X_train, y_train)


# Add back seasonal component for test period
seasonal_test = df['seasonal'].iloc[-len(X_test):].values
#dt_pred = dt_pred_sa + seasonal_test

# Predict on train and test sets
train_pred_sa = dt_model.predict(X_train)
test_pred_sa = dt_model.predict(X_test)

# Calculate residuals on training set
y_train = np.array(y_train).flatten()
train_pred_sa = np.array(train_pred_sa).flatten()
residuals_train = y_train - train_pred_sa

# Find aligned DateTime index for training residuals
datetime_index_train = df['DateTime'].iloc[:len(y_train)].reset_index(drop=True)

# Create pandas Series for residuals with datetime index
residuals_series = pd.Series(residuals_train, index=pd.to_datetime(datetime_index_train)).asfreq('3h').dropna()

# Fit ETS model on residuals with additive trend and seasonality
ets_model = ExponentialSmoothing(residuals_series, trend='add', seasonal='add', seasonal_periods=8).fit()

# Forecast residuals for test duration
residuals_forecast = ets_model.forecast(len(y_test))

# Add back seasonal component to test predictions and ETS residual forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast.values

# Calculate Mean Absolute Scaled Error
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Compute metrics
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
r2 = r2_score(y_test, hybrid_forecast)

print("STL + Decision Tree + ETS Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
###########STL+DT+SARIMAX###########
# Fit SARIMAX on residuals (order and seasonal_order can be tuned)
sarimax_model = SARIMAX(residuals_series, order=(0,0,1), seasonal_order=(0,0,0,0))
sarimax_fit = sarimax_model.fit(disp=False)

# Forecast residuals for test period length
residuals_forecast = sarimax_fit.get_forecast(steps=len(y_test)).predicted_mean

# Add back seasonal component for test period
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Compose hybrid forecast with seasonal + DT predictions + residual forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast.values

# Define Mean Absolute Scaled Error (MASE)
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Metrics computation
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
r2 = r2_score(y_test, hybrid_forecast)

print("STL + Decision Tree + SARIMAX Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
########STL+DT+TBATS########
import pandas as pd
import numpy as np
from tbats import TBATS
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming STL and feature engineering and train-test split code as before

# Train Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on train and test sets
train_pred_sa = dt_model.predict(X_train)
test_pred_sa = dt_model.predict(X_test)

# Convert to numpy arrays and flatten for residuals
y_train_arr = np.array(y_train).flatten()
train_pred_arr = np.array(train_pred_sa).flatten()

# Calculate residuals on training set
residuals_train = y_train_arr - train_pred_arr

# Get aligned datetime index from train samples
datetime_index_train = df['DateTime'].iloc[:len(y_train_arr)].reset_index(drop=True)

# Create residuals time series
residuals_series = pd.Series(residuals_train, index=pd.to_datetime(datetime_index_train)).asfreq('3h').dropna()

# Downsample residuals if needed to reduce TBATS memory usage
residuals_downsampled = residuals_series.asfreq('6h').dropna()

# Fit TBATS model
tbats_model = TBATS(seasonal_periods=[4], use_arma_errors=False,
                    use_box_cox=False, use_damped_trend=False, n_jobs=1)
tbats_fit = tbats_model.fit(residuals_downsampled)

# Forecast residuals for test length (adjust if downsampled)
steps = len(y_test)  # Or adjust if you downsample test set too
residuals_forecast = tbats_fit.forecast(steps=steps)

# Add back seasonal component and residual forecast to test predictions
seasonal_test = df['seasonal'].iloc[-len(y_test):].values
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast

# MASE definition
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n-1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Metrics
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train_arr)
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
r2 = r2_score(y_test, hybrid_forecast)

print("STL + Decision Tree + TBATS Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
########STL+RF+ETS#####
import pandas as pd
from statsmodels.tsa.seasonal import STL
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assume you have already loaded and pre-processed your data as in your previous scripts
# Including STL decomposition, one-hot encoding, lag features, and train-test split
# Load data
df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data.xlsx')
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
df = df.set_index('DateTime')
df = df.asfreq('3h')  # Assuming 3-hour frequency

# STL decomposition with daily seasonality (period=8)
stl = STL(df['DI'], period=8)
res = stl.fit()

df['seasonal'] = res.seasonal
df['trend'] = res.trend
df['DI_sa'] = df['DI'] - df['seasonal']  # Seasonally adjusted series
df.dropna(inplace=True)  # Remove NaNs created by STL edges

# Reset index for feature engineering
df.reset_index(inplace=True)

# Extract datetime features
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month

# One-hot encode Season column from original data
original_df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data.xlsx')
original_df['DateTime'] = pd.to_datetime(original_df['DateTime'], errors='coerce')
# Align and add Season column
df['Season'] = original_df.set_index('DateTime').loc[df['DateTime'], 'Season'].values
df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Lag features on seasonally adjusted DI
df['DI_sa_lag_1'] = df['DI_sa'].shift(1)
df['DI_sa_lag_3'] = df['DI_sa'].shift(3)
df['DI_sa_lag_6'] = df['DI_sa'].shift(6)

df.dropna(inplace=True)

from sklearn.model_selection import train_test_split

features = ['Rainfall', 'Cloud', 'Atmosphere', 'Wind', 'Hour', 'Day', 'Month',
            'DI_sa_lag_1', 'DI_sa_lag_3', 'DI_sa_lag_6',
            'Season_Spring', 'Season_Summer', 'Season_Winter']

X = df[features]
y = df['DI_sa']  # seasonally adjusted target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest on seasonally adjusted series
rf_model = RandomForestRegressor(
    n_estimators=100,          # number of trees
    max_depth=6,               # maximum depth of each tree (reduce complexity)
    min_samples_leaf=50,       # minimum samples required at leaf nodes
    max_features='sqrt',       # number of features considered at each split
    random_state=42,
    n_jobs=-1                  # use all cores
)
#rf_model = RandomForestRegressor(n_estimators=100, random_state=702)
rf_model.fit(X_train, y_train)

# Predict on train and test sets
train_pred_sa = rf_model.predict(X_train)
test_pred_sa = rf_model.predict(X_test)

# Compute residuals on training set
y_train_arr = np.array(y_train).flatten()
train_pred_arr = np.array(train_pred_sa).flatten()
residuals_train = y_train_arr - train_pred_arr

# Prepare DateTime index for residuals
datetime_index_train = df['DateTime'].iloc[:len(y_train_arr)].reset_index(drop=True)

# Create residuals time series without 'asfreq'
residuals_series = pd.Series(residuals_train, index=pd.to_datetime(datetime_index_train))

# Fit ETS on residuals
ets_model = ExponentialSmoothing(residuals_series, trend='add', seasonal='add', seasonal_periods=8).fit()

# Forecast residuals for test set length
residuals_forecast = ets_model.forecast(len(y_test))

# Add back seasonal component for test set
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Hybrid forecast: base prediction + seasonal + residual forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast.values

# Define MASE
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Metrics
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
r2 = r2_score(y_test, hybrid_forecast)

print("STL + Random Forest + ETS Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
#####STL+RF+SARIMAX######


# Fit SARIMAX on residuals (use a basic SARIMA config)
sarimax_model = SARIMAX(residuals_series, order=(0,0,1), seasonal_order=(0,0,0,0))
sarimax_fit = sarimax_model.fit(disp=False)

# Forecast residuals for test set length
residuals_forecast = sarimax_fit.get_forecast(steps=len(y_test)).predicted_mean

# Add back seasonal component for test set
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Hybrid forecast: base prediction + seasonal + residual forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast.values

# Define MASE
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Metrics
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
r2 = r2_score(y_test, hybrid_forecast)

print("STL + Random Forest + SARIMAX Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
########STL+RF+TBATS###########
# Downsample residuals to 6h frequency to reduce memory usage in TBATS
residuals_downsampled = residuals_series.asfreq('6h').dropna()

# Fit TBATS
tbats_model = TBATS(seasonal_periods=[4], use_arma_errors=False,
                    use_box_cox=False, use_damped_trend=False, n_jobs=1)
tbats_fit = tbats_model.fit(residuals_downsampled)

# Forecast residuals length adjusted for downsampling
steps = len(y_test)  # if y_test is 3h freq, adjust accordingly if needed
residuals_forecast = tbats_fit.forecast(steps=steps)

# Add back seasonal component and residual forecast
seasonal_test = df['seasonal'].iloc[-len(y_test):].values
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast

# MASE calculation function
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Metrics
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train_arr)
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
r2 = r2_score(y_test, hybrid_forecast)

print("STL + Random Forest + TBATS Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
########STL+FP+ETS##########
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data.xlsx')
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
df = df.set_index('DateTime')
df = df.asfreq('3h')  # Assuming 3-hour frequency

# STL decomposition with daily seasonality (period=8)
stl = STL(df['DI'], period=8)
res = stl.fit()

df['seasonal'] = res.seasonal
df['trend'] = res.trend
df['DI_sa'] = df['DI'] - df['seasonal']  # Seasonally adjusted series
df.dropna(inplace=True)  # Remove NaNs created by STL edges

# Reset index for feature engineering
df.reset_index(inplace=True)

# Extract datetime features
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month

# One-hot encode Season column from original data
original_df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data.xlsx')
original_df['DateTime'] = pd.to_datetime(original_df['DateTime'], errors='coerce')
# Align and add Season column
df['Season'] = original_df.set_index('DateTime').loc[df['DateTime'], 'Season'].values
df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Lag features on seasonally adjusted DI
df['DI_sa_lag_1'] = df['DI_sa'].shift(1)
df['DI_sa_lag_3'] = df['DI_sa'].shift(3)
df['DI_sa_lag_6'] = df['DI_sa'].shift(6)

df.dropna(inplace=True)

from sklearn.model_selection import train_test_split

features = ['Rainfall', 'Cloud', 'Atmosphere', 'Wind', 'Hour', 'Day', 'Month',
            'DI_sa_lag_1', 'DI_sa_lag_3', 'DI_sa_lag_6',
            'Season_Spring', 'Season_Summer', 'Season_Winter']

X = df[features]
y = df['DI_sa']  # seasonally adjusted target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


from prophet import Prophet

df_prophet = df[['DateTime', 'DI_sa', 'Rainfall', 'Cloud', 'Atmosphere', 'Wind', 'Hour', 'Day', 'Month',
                 'DI_sa_lag_1', 'DI_sa_lag_3', 'DI_sa_lag_6',
                 'Season_Spring', 'Season_Summer', 'Season_Winter']].copy()
df_prophet.rename(columns={'DateTime': 'ds', 'DI_sa': 'y'}, inplace=True)

train_df = df_prophet.iloc[:len(X_train)]
test_df = df_prophet.iloc[len(X_train):]

prophet_model = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False)
for reg in features:
    prophet_model.add_regressor(reg)

prophet_model.fit(train_df[['ds', 'y'] + features])
future = prophet_model.make_future_dataframe(periods=len(test_df), freq='3h')
future = future.merge(df_prophet[features + ['ds']], on='ds', how='left')
forecast = prophet_model.predict(future)




# Predict on train and test sets (just seasonally adjusted)
train_pred_sa = forecast['yhat'].iloc[:len(train_df)].values
test_pred_sa = forecast['yhat'].iloc[-len(test_df):].values

# Residuals for train set
y_train_arr = np.array(y_train).flatten()
residuals_train = y_train_arr - train_pred_sa

# ETS on residuals
datetime_index_train = train_df['ds'].reset_index(drop=True)
residuals_series = pd.Series(residuals_train, index=pd.to_datetime(datetime_index_train)).asfreq('3h')

ets_model = ExponentialSmoothing(residuals_series, trend='add', seasonal='add', seasonal_periods=8).fit()
residuals_forecast = ets_model.forecast(len(y_test))

# Seasonal component for test set
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Hybrid forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast.values

# MASE
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Metrics
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
epsilon = 1e-8
mape = np.mean(np.abs((y_test - hybrid_forecast) / (y_test + epsilon))) * 100
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
r2 = r2_score(y_test, hybrid_forecast)

print("STL + FB Prophet + ETS Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
############STL+FP+SARIMAX#####
# Fit SARIMAX to residuals
sarimax_model = SARIMAX(residuals_series, order=(0,0,1), seasonal_order=(0,0,0,0))
sarimax_fit = sarimax_model.fit(disp=False)

# Forecast residuals for test set duration
residuals_forecast = sarimax_fit.get_forecast(steps=len(y_test)).predicted_mean

# Get seasonal component for test set
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Hybrid forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast

# Evaluation
import numpy as np

# Convert to numpy arrays
y_true = np.array(y_test)
y_pred = np.array(hybrid_forecast)

# Calculate MAPE, avoiding division by zero
# Set a small epsilon
epsilon = 1e-8

# replace zeros in y_true temporarily for division
denominator = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
mape_value = np.mean(np.abs((y_true - y_pred) / denominator)) * 100

print(f"Corrected MAPE: {mape_value} %")


# Compute the denominator for MASE, adding small epsilon to avoid zero
y_test_array = np.array(y_test)
hybrid_forecast_array = np.array(hybrid_forecast)

denominator = np.abs(np.diff(y_train)).mean()
epsilon = 1e-8
if denominator < epsilon:
    mase = np.nan  # or assign a large value, or skip
else:
    errors = np.abs(y_test_array - hybrid_forecast_array)
    mase = errors.mean() / denominator

# Print results
print("Corrected MASE:", mase)



mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
r2 = r2_score(y_test, hybrid_forecast)

print("STL + FB Prophet + SARIMAX Hybrid Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")
###########STL+FP+TBATS############
# Downsample residuals to reduce TBATS memory usage
residuals_downsampled = residuals_series.asfreq('6h').dropna()

# Fit TBATS
tbats_model = TBATS(seasonal_periods=[4], use_arma_errors=False,
                    use_box_cox=False, use_damped_trend=False, n_jobs=1)
tbats_fit = tbats_model.fit(residuals_downsampled)

# Forecast residuals for test length
steps = len(test_df)  # Adjust if needed according to freq
residuals_forecast = tbats_fit.forecast(steps=steps)

# Seasonal component for test set
seasonal_test = df['seasonal'].iloc[-len(test_df):].values

# Hybrid forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast

# Metrics functions
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
r2 = r2_score(y_test, hybrid_forecast)

print("STL + FB Prophet + TBATS Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
############STL+SVR+ETS####

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Assume STL, feature engineering, and train-test split are done as per your previous code

# SVR pipeline with imputation and scaling
svr_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale"))
])
svr_pipe.fit(X_train, y_train)

# Predictions
train_pred_sa = svr_pipe.predict(X_train)
test_pred_sa = svr_pipe.predict(X_test)

# Compute residuals on train data
y_train_arr = np.array(y_train).flatten()
train_pred_arr = np.array(train_pred_sa).flatten()
residuals_train = y_train_arr - train_pred_arr

# Create residuals time series with aligned DateTime index
datetime_index_train = df['DateTime'].iloc[:len(y_train_arr)].reset_index(drop=True)
residuals_series = pd.Series(residuals_train, index=pd.to_datetime(datetime_index_train))

# Fit ETS model on residuals (additive trend and seasonality)
ets_model = ExponentialSmoothing(residuals_series, trend='add', seasonal='add', seasonal_periods=8).fit()

# Forecast residuals for test period
residuals_forecast = ets_model.forecast(len(y_test))

# Seasonal component for test set
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Hybrid forecast by adding SVR test prediction, seasonal, and residual forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast.values

# MASE metric function
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Metrics
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
r2 = r2_score(y_test, hybrid_forecast)

print("STL + SVR + ETS Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
####STL+SVR+SARIMAX###########
# Fit SARIMAX residual model
sarimax_model = SARIMAX(residuals_series, order=(0,0,1), seasonal_order=(0,0,0,0))
sarimax_fit = sarimax_model.fit(disp=False)

# Forecast residuals for test period
residuals_forecast = sarimax_fit.get_forecast(steps=len(y_test)).predicted_mean

# Seasonal component for test set
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Hybrid forecast combining base prediction, seasonal component, and residual forecasts
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast.values

# Define MASE calculation
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n -1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Compute evaluation metrics
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
r2 = r2_score(y_test, hybrid_forecast)

print("STL + SVR + SARIMAX Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
#####STL+SVR+TBATS############
# Downsample residuals for TBATS to reduce memory use (optional but recommended)
residuals_downsampled = residuals_series.asfreq('6h').dropna()

# Fit TBATS model on residuals
tbats_model = TBATS(seasonal_periods=[4], use_arma_errors=False,
                    use_box_cox=False, use_damped_trend=False, n_jobs=1)
tbats_fit = tbats_model.fit(residuals_downsampled)

# Forecast residuals for test length (match length carefully if downsampling was done)
steps = len(y_test)  # Adjust if you downsample test as well
residuals_forecast = tbats_fit.forecast(steps=steps)

# Retrieve seasonal component for test samples
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Compose final hybrid forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast

# Define MASE metric
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Calculate metrics
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train_arr)
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
r2 = r2_score(y_test, hybrid_forecast)

print("STL + SVR + TBATS Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
###########STL+XGBOOST+ETS####

from xgboost import XGBRegressor

# Assume STL decomposition, feature engineering, and train-test split are done as you previously did

# Train XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on train and test
train_pred_sa = xgb_model.predict(X_train)
test_pred_sa = xgb_model.predict(X_test)

# Calculate residuals on training set
y_train_arr = np.array(y_train).flatten()
train_pred_arr = np.array(train_pred_sa).flatten()
residuals_train = y_train_arr - train_pred_arr

# Prepare pandas Series for residuals with DateTime index
datetime_index_train = df['DateTime'].iloc[:len(y_train_arr)].reset_index(drop=True)
residuals_series = pd.Series(residuals_train, index=pd.to_datetime(datetime_index_train))

# Fit ETS model on residuals
ets_model = ExponentialSmoothing(residuals_series, trend='add', seasonal='add', seasonal_periods=8).fit()

# Forecast residuals for test length
residuals_forecast = ets_model.forecast(len(y_test))

# Add back seasonal component for test set
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Compose hybrid forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast.values

# Define MASE metric function
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n-1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom

# Evaluation metrics
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
mape = np.mean(np.abs((y_test - hybrid_forecast) / y_test)) * 100
mase = mean_absolute_scaled_error(y_test, hybrid_forecast, y_train)
r2 = r2_score(y_test, hybrid_forecast)

print("STL + XGBoost + ETS Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
#############STL+XGBoost+SARIMAX####

# Fit SARIMAX model on residuals
sarimax_model = SARIMAX(residuals_series, order=(0,0,1), seasonal_order=(0,0,0,0))
sarimax_fit = sarimax_model.fit(disp=False)

# Forecast residuals for test set length
residuals_forecast = sarimax_fit.get_forecast(steps=len(y_test)).predicted_mean

# Seasonal component for test
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Hybrid forecast: base + seasonal + residual correction
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast.values

# Corrected MAPE calculation to avoid division by zero
epsilon = 1e-8
mape = np.mean(np.abs((y_test - hybrid_forecast) / np.where(np.abs(y_test) < epsilon, epsilon, np.abs(y_test)))) * 100

# Corrected MASE calculation with denominator check
denominator = np.abs(np.diff(y_train_arr)).mean()
if denominator < epsilon:
    mase = np.nan
else:
    errors = np.abs(y_test - hybrid_forecast)
    mase = errors.mean() / denominator

# Other metrics
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
r2 = r2_score(y_test, hybrid_forecast)

# Print metrics
print("STL + XGBoost + SARIMAX Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
######STL+XGBoost+TBATS###


# Downsample residuals for TBATS to reduce memory
residuals_downsampled = residuals_series.asfreq('6h').dropna()

# Fit TBATS
tbats_model = TBATS(seasonal_periods=[8], use_arma_errors=False,
                    use_box_cox=False, use_damped_trend=False, n_jobs=1)
tbats_fit = tbats_model.fit(residuals_downsampled)

# Forecast residuals for test set length (adjust if downsampling)
steps = len(y_test)  # Adjust if you downsample test set
residuals_forecast = tbats_fit.forecast(steps=steps)

# Seasonal component for test set
seasonal_test = df['seasonal'].iloc[-len(y_test):].values

# Hybrid forecast
hybrid_forecast = test_pred_sa + seasonal_test + residuals_forecast

# Corrected MAPE for zero division
epsilon = 1e-8
mape = np.mean(np.abs((y_test - hybrid_forecast) / np.where(np.abs(y_test) < epsilon, epsilon, np.abs(y_test)))) * 100

# MASE denominator check
denominator = np.abs(np.diff(y_train_arr)).mean()
if denominator < epsilon:
    mase = np.nan
else:
    errors = np.abs(y_test - hybrid_forecast)
    mase = errors.mean() / denominator

# Other metrics
mse = mean_squared_error(y_test, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, hybrid_forecast)
r2 = r2_score(y_test, hybrid_forecast)

print("STL + XGBoost + TBATS Hybrid Model Metrics:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"MASE: {mase}")
print(f"R2: {r2}")
####STL+LSTM+ETS###
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load data and STL decomposition
df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data_LSTM.xlsx')
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
df = df.set_index('DateTime').asfreq('3h')

stl = STL(df['DI'], period=8)
res = stl.fit()
df['seasonal'] = res.seasonal
df['trend'] = res.trend
df['DI_sa'] = df['DI'] - df['seasonal']
df.dropna(inplace=True)
df.reset_index(inplace=True)

# Extract date/time features
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month

# One-hot encode Season column aligned from original data
original_df = pd.read_excel('C:/Users/rawfu/OneDrive/Desktop/Thesis/DI_data_LSTM.xlsx')
original_df['DateTime'] = pd.to_datetime(original_df['DateTime'], errors='coerce')
df['Season'] = original_df.set_index('DateTime').loc[df['DateTime'], 'Season'].values
df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Create lag and rolling features on seasonally adjusted DI
df['DI_sa_lag_1'] = df['DI_sa'].shift(1)
df['DI_sa_lag_3'] = df['DI_sa'].shift(3)
df['DI_sa_lag_6'] = df['DI_sa'].shift(6)
df['rollmean_7'] = df['DI_sa'].rolling(window=7).mean()
df['rollstd_7'] = df['DI_sa'].rolling(window=7).std()
df.dropna(inplace=True)

# Define features and target (next-step prediction)
feature_cols = ['DI_sa', 'Rainfall', 'Cloud', 'Atmosphere', 'Wind', 'Hour', 'Day', 'Month',
                'DI_sa_lag_1', 'DI_sa_lag_3', 'DI_sa_lag_6',
                'rollmean_7', 'rollstd_7',
                'Season_Spring', 'Season_Summer', 'Season_Winter']

X = df[feature_cols]
Y = df['DI_sa'].shift(-1).dropna()
X = X.iloc[:-1, :]
seasonal_test = df['seasonal'].iloc[-len(Y):].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.2, shuffle=False)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create sequences for LSTM
time_steps = 6

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, time_steps)
seasonal_test_seq = seasonal_test[time_steps:]

# Build LSTM model
model_lstm = Sequential()
model_lstm.add(Input(shape=(time_steps, X_train_seq.shape[2])))
model_lstm.add(LSTM(units=50, return_sequences=True))
model_lstm.add(Dropout(0.3))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model_lstm.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test_seq), callbacks=[early_stop], verbose=2)

# Residuals calculation on training set
train_pred = model_lstm.predict(X_train_seq).flatten()
residuals_train = y_train_seq - train_pred

# ETS model to residuals using exact train datetime index

datetime_index_train = df['DateTime'].iloc[:len(y_train_seq)].reset_index(drop=True)
residuals_series = pd.Series(residuals_train, index=pd.to_datetime(datetime_index_train)).asfreq('3h').dropna()

ets_model = ExponentialSmoothing(residuals_series, trend='add', seasonal='add', seasonal_periods=8).fit()

# Forecast residuals for test length
residuals_forecast = ets_model.forecast(len(y_test_seq))

# Predict on test set
y_pred_sa = model_lstm.predict(X_test_seq).flatten()

# Hybrid forecast = LSTM seasonally adjusted preds + ETS residual forecast + seasonal component

# Get date range for test sequences (after time_steps offset)
datetime_test_seq = df['DateTime'].iloc[-len(y_test_seq):].reset_index(drop=True)

# Slice seasonal_test using these dates
seasonal_test_seq_aligned = df.set_index('DateTime').loc[datetime_test_seq, 'seasonal'].values

import pandas as pd

seasonal_series = pd.Series(seasonal_test, index=pd.date_range(start=datetime_test_seq.min(), periods=len(seasonal_test), freq='3h'))

seasonal_test_seq_aligned = seasonal_series.loc[datetime_test_seq].values

hybrid_forecast = y_pred_sa + residuals_forecast.values + seasonal_test_seq_aligned

# Metrics calculation
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n -1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

mse = mean_squared_error(y_test_seq, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_seq, hybrid_forecast)
mape = np.mean(np.abs((y_test_seq - hybrid_forecast) / y_test_seq)) * 100
mase = mean_absolute_scaled_error(y_test_seq, hybrid_forecast, y_train_seq)
r2 = r2_score(y_test_seq, hybrid_forecast)

print("STL + LSTM + ETS Hybrid Model Metrics:")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"MAPE: {mape:.6f}%")
print(f"MASE: {mase:.6f}")
print(f"R²: {r2:.6f}")

####STL+LSTM+SARIMAX###
# Fit SARIMAX on residuals


from statsmodels.tsa.statespace.sarimax import SARIMAX

sarimax_model = SARIMAX(residuals_series, order=(0,0,1), seasonal_order=(0,0,0,0))
sarimax_fit = sarimax_model.fit(disp=False)

# Forecast residuals for test sequences length
residuals_forecast = sarimax_fit.get_forecast(steps=len(y_test_seq)).predicted_mean

# Predict LSTM on test sequences
y_pred_sa = model_lstm.predict(X_test_seq).flatten()

# Compose hybrid forecast
hybrid_forecast = y_pred_sa + seasonal_test_seq_aligned + residuals_forecast.values

# Eval metrics
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n-1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom
mase = mean_absolute_scaled_error(y_test_seq, hybrid_forecast, y_train_seq)
mse = mean_squared_error(y_test_seq, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_seq, hybrid_forecast)
mape = np.mean(np.abs((y_test_seq - hybrid_forecast) / y_test_seq)) * 100
r2 = r2_score(y_test_seq, hybrid_forecast)

print("STL + LSTM + SARIMAX Hybrid Model Metrics:")
print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}%, MASE: {mase:.6f}, R²: {r2:.6f}")

####STL+LSTM+TBATS###
# Fit TBATS model on residuals
from tbats import TBATS

tbats_model = TBATS(seasonal_periods=[8])
tbats_fit = tbats_model.fit(residuals_series)

# Forecast residuals for test length
residuals_forecast = tbats_fit.forecast(steps=len(y_test_seq))

hybrid_forecast = y_pred_sa + seasonal_test_seq_aligned + residuals_forecast


# Eval metrics
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    n = len(y_train)
    denom = np.abs(np.diff(y_train)).sum() / (n-1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / denom
mase = mean_absolute_scaled_error(y_test_seq, hybrid_forecast, y_train_seq)
mse = mean_squared_error(y_test_seq, hybrid_forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_seq, hybrid_forecast)
mape = np.mean(np.abs((y_test_seq - hybrid_forecast) / y_test_seq)) * 100
r2 = r2_score(y_test_seq, hybrid_forecast)

print("STL + LSTM + TBATS Hybrid Model Metrics:")
print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}%, MASE: {mase:.6f}, R²: {r2:.6f}")

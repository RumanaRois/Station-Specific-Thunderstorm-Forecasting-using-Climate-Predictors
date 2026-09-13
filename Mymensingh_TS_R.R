# Load Necessary Packages
library(forecast)
library(e1071)
library(xgboost)
library(tensorflow)
library(keras3)
library(seasonal)
library(Metrics)
library(ggplot2)
library(dplyr)
library(lubridate)
library(tidyr)
library(reshape2)
library(fpp2)
library(patchwork)

# Data sets
setwd("D:/Mahin and Amrin Papers/Paper Mam/2. Clean Data")
D1 = read.csv("Mymensingh_MTS_Data.csv")

# Time Series Data
ts1 = ts(D1$MT, frequency = 12, start = c(1985, 1))
head(ts1)

# Time Series Plot
D1$Date = seq(as.Date("1985-01-01"), by = "month", length.out = nrow(D1))

ggplot(D1, aes(x = Date, y = MT)) +
  geom_line(color = "steelblue", size = 1) +
  labs(
    x = "Year",
    y = "Monthly Total TS Days in Sreemangal"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.border = element_rect(fill = NA, colour = "black", linewidth = 1)
  )

# STL Decomposition
STL = stl(ts1, s.window = 5)
plot(STL)
STL_trend = trendcycle(STL)
STL_seasonal = seasonal(STL)
STL_Sadj = seasadj(STL)

# Add the seasonal-adjusted and trend data to the data frame
D1$Seasonally_Adjusted = as.numeric(STL_Sadj)
D1$Trend = as.numeric(STL_trend)

# Reshape the data for ggplot2
data_long = D1 |> 
  pivot_longer(
    cols = c(MT, Seasonally_Adjusted, Trend),
    names_to = "Component",
    values_to = "Value"
  )

# Time Series Plot with Legend
p1 = ggplot(data_long, aes(x = Date, y = Value, color = Component)) +
  geom_line(linewidth = 1, alpha = 1) +
  scale_color_manual(
    values = c(
      "MT" = "#50C878",         # Blue for original data
      "Seasonally_Adjusted" = "red", # Orange for seasonal adjustment
      "Trend" = "steelblue"               # Green for trend
    ),
    labels = c(
      "MT" = "Monthly TS Days",
      "Seasonally_Adjusted" = "Seasonal Adjusted",
      "Trend" = "Trend"
    )
  ) +
  labs(
    x = "Year",
    y = "Monthly Total TS Days in Mymensingh",
    color = "Components",
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 11),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
  )


print(p1)


# Box plot
D1$M = factor(D1$M, labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))

p2 = ggplot(D1, aes(x = M, y = MT, fill = M)) +
  geom_boxplot(color = "red") +
  labs(x = "Month",
       y = "Monthly Total TS Days in Mymensingh"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    panel.border = element_rect(fill = NA, colour = "black", linewidth = 1),
    axis.text.x = element_text(angle = 360, hjust = 1)
  )
library(patchwork)
#tiff("Figure 4.tiff", width = 9, height = 4, units = "in", res = 300)
#p1 + p2
#dev.off()

# Heat Map
ggplot(D1, aes(x = M, y = factor(Y), fill = MT)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "darkblue") +
  labs(x = "Month", y = "Year") +
  theme_minimal() + 
  theme(
    legend.title  = element_blank(),
    panel.border = element_rect(fill = NA, colour = "black", linewidth = 1)
  )

# Train Test Split
m = length(ts1)
train_length = round(m*0.8)

train1 = ts(ts1[1:train_length], frequency = 12, start = c(1985,1))
test1 = ts(ts1[(train_length + 1):m], frequency = 12, start = c(2017,1))
k = length(test1)

# 1. ARIMA
fit1 = auto.arima(train1)
F_ARIMA = forecast(fit1, h = k)$mean


# 2. ETS
fit2 = ets(train1)
F_ETS = forecast(fit2, h = k)$mean


# 3. ANN
fit3 = nnetar(train1)
F_ANN = forecast(fit3, h = k)$mean


# 4. SVR
X1 = D1$MT[1:train_length]
Y1 = D1$MT[2:(train_length + 1)]
X2 = D1$MT[train_length:(m-1)]
Y2 = D1$MT[(train_length + 1):m]

fit4 = svm(X1, Y1)

F_SVR = as.numeric(predict(fit4, X2))
F_SVR = ts(F_SVR, frequency = 12, start = c(2017,1))


# 5. XGBoost
prepare_data_xgb = function(d, n_steps){
  X = list()
  y = list()
  for(i in 1:(length(d) - n_steps)){
    end_ix = i + n_steps - 1
    seq_x = d[i:end_ix]
    seq_y = d[end_ix + 1]
    X = c(X, list(seq_x))
    y = c(y, seq_y)
  }
  X = do.call(rbind, X)
  y = unlist(y)
  return(list(X = X, y = y))
}

train = D1$MT[1:(train_length - 12)]
test = D1$MT[((train_length - 12) + 1):m]
n_steps = 12


# Prepare training set
data_train = prepare_data_xgb(train, n_steps)
X_train = data_train$X
y_train = data_train$y

# Prepare testing set
data_test = prepare_data_xgb(test, n_steps)
X_test = data_test$X
y_test = data_test$y

# === 2. Train XGBoost model ===

# Convert to xgbDMatrix
dtrain1 = xgb.DMatrix(data = X_train, label = y_train)
dtest1 = xgb.DMatrix(data = X_test, label = y_test)

# Set XGBoost parameters
params = list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train model
fit5 = xgb.train(
  params = params,
  data = dtrain1,
  nrounds = 100,
  watchlist = list(train = dtrain1, test = dtest1),
  early_stopping_rounds = 10,
  verbose = 1
)

# === 3. Make Predictions ===
F_XGB = predict(fit5, dtest1)
F_XGB = ts(F_XGB, frequency = 12, start = c(2017,1))


# 6.LSTM

# Environment Set up

library(reticulate)
#conda_list()
#use_condaenv("keras-tf", required = TRUE)
#py_config()
# conda_install(envname = "keras-tf", packages = "tensorflow", pip = TRUE)
library(tensorflow)
library(keras3)
#tf$constant("Hello from TensorFlow!")

# Set Up Data
prepare_data = function(d, n_steps){
  X = list()
  y = list()
  for(i in 1:length(d)){
    end_ix = i + n_steps - 1
    if(end_ix > length(d) - 1){
      break
    }
    seq_x = d[i:end_ix]
    seq_y = d[end_ix + 1]
    X = c(X, list(seq_x))
    y = c(y, seq_y)
  }
  return(list(X = X, y = y))
}

n_steps = 12
n_features = 1

# Define Model
model = keras_model_sequential()
model$add(layer_lstm(units = 50, activation = "relu", input_shape = c(n_steps, n_features), return_sequences = FALSE))
model$add(layer_dense(units = 1))

# Compile Model
model$compile(
  optimizer = "adam",
  loss = "mse"
)

# Train-test splitting
X_train = prepare_data(train, n_steps)$X
X_train = array(as.numeric(unlist(X_train)), dim = c(length(X_train), n_steps, 1))

y_train = as.numeric(prepare_data(train, n_steps)$y)

X_test = prepare_data(test, n_steps)$X
X_test = array(as.numeric(unlist(X_test)), dim = c(length(X_test), n_steps, 1))

y_test = as.numeric(prepare_data(test, n_steps)$y)

X_train_np = np_array(X_train)
y_train_np = np_array(y_train)
X_test_np = np_array(X_test)

# Fit model
fit6 = model$fit(
  X_train_np,
  y_train_np,
  batch_size = as.integer(12),
  epochs = as.integer(15),
  verbose = 2L
)

length(y_test)

# Make predictions
F_LSTM = model$predict(X_test_np)
F_LSTM = ts(F_LSTM, frequency = 12, start = c(2017,1))


################# X11 Decomposition
SeasonaX11 = seas(ts1, x11 = "")
plot(SeasonaX11)
Sea_X11 = seasonal(SeasonaX11)
Adj_X11 = seasadj(SeasonaX11)

# Model Using X11
X11_model = seas(train1, x11 = "")
Adj_X11_train1 = seasadj(X11_model)

# 1. X11 + ARIMA
M1 = auto.arima(Adj_X11_train1)
F_X11_ARIMA = forecast::forecast(M1, h = k)$mean + Sea_X11[(train_length + 1):m]

# 2. X11 + ETS
M2 = ets(Adj_X11_train1)
F_X11_ETS = forecast::forecast(M2, h = k)$mean + Sea_X11[(train_length + 1):m]

# 3. X11 + ANN
M3 = nnetar(Adj_X11_train1)
F_X11_ANN = forecast::forecast(M3, h = k)$mean + Sea_X11[(train_length + 1):m]

# 4. X11 + SVR
X11 = Adj_X11[1:train_length]
Y11 = Adj_X11[2:(train_length + 1)]
X21 = Adj_X11[train_length:(m-1)]
Y21 = Adj_X11[(train_length + 1):m]

M4 = svm(X11, Y11)

F_X11_SVR = as.numeric(predict(M4, X21)) + Sea_X11[(train_length + 1):m]
F_X11_SVR = ts(F_X11_SVR, frequency = 12, start = c(2017,1))


# 5. X11 + XGB
X11_train = Adj_X11[1:(train_length - 12)]
X11_test = Adj_X11[((train_length - 12) + 1):m]
n_steps = 12

# Prepare training set
data_train = prepare_data_xgb(X11_train, n_steps)
X11_X_train = data_train$X
X11_y_train = data_train$y

# Prepare testing set
data_test = prepare_data_xgb(X11_test, n_steps)
X11_X_test = data_test$X
X11_y_test = data_test$y

# === 2. Train XGBoost model ===

# Convert to xgbDMatrix
dtrain2 = xgb.DMatrix(data = X11_X_train, label = X11_y_train)
dtest2 = xgb.DMatrix(data = X11_X_test, label = X11_y_test)

# Train model
M5 = xgb.train(
  params = params,
  data = dtrain2,
  nrounds = 100,
  watchlist = list(train = dtrain2, test = dtest2),
  early_stopping_rounds = 10,
  verbose = 1
)

# === 3. Make Predictions ===
F_X11_XGB = predict(M5, dtest2) + Sea_X11[(train_length + 1):m]
F_X11_XGB = ts(F_X11_XGB, frequency = 12, start = c(2017,1))


# X11 + LSTM

# Define Model
model_2 = keras_model_sequential()
model_2$add(layer_lstm(units = 50, activation = "relu", input_shape = c(n_steps, n_features), return_sequences = FALSE))
model_2$add(layer_dense(units = 1))

# Compile Model
model_2$compile(
  optimizer = "adam",
  loss = "mse"
)

X_train = prepare_data(X11_train, n_steps)$X
X_train = array(as.numeric(unlist(X_train)), dim = c(length(X_train), n_steps, 1))

y_train = as.numeric(prepare_data(X11_train, n_steps)$y)

X_test = prepare_data(X11_test, n_steps)$X
X_test = array(as.numeric(unlist(X_test)), dim = c(length(X_test), n_steps, 1))

y_test = as.numeric(prepare_data(X11_test, n_steps)$y)

X_train_np = np_array(X_train)
y_train_np = np_array(y_train)

# Fit model
M6 = model_2$fit(
  X_train_np,
  y_train_np,
  batch_size = as.integer(12),
  epochs = as.integer(15),
  verbose = 2L
)


# Make predictions
F_X11_LSTM = model_2$predict(X_test) + Sea_X11[(train_length + 1):m]
F_X11_LSTM = ts(F_X11_LSTM, frequency = 12, start = c(2017,1))


################# STL Decomposition
SeasonaSTL = stl(ts1, s.window = 5)
plot(SeasonaSTL)
Sea_STL = seasonal(SeasonaSTL)
Adj_STL = seasadj(SeasonaSTL)

# Model Using X11
STL_model = stl(train1, s.window = 5)
Adj_STL_train1 = seasadj(STL_model)

# 1. STL + ARIMA
S1 = auto.arima(Adj_STL_train1)
F_STL_ARIMA = forecast::forecast(S1, h = k)$mean + Sea_STL[(train_length + 1):m]

# 2. STL + ETS
S2 = ets(Adj_STL_train1)
F_STL_ETS = forecast::forecast(S2, h = k)$mean + Sea_STL[(train_length + 1):m]

# 3. STL + ANN
S3 = nnetar(Adj_STL_train1)
F_STL_ANN = forecast::forecast(S3, h = k)$mean + Sea_STL[(train_length + 1):m]

# 4. STL + SVR
X12 = Adj_STL[1:train_length]
Y12 = Adj_STL[2:(train_length + 1)]
X22 = Adj_STL[train_length:(m-1)]
Y22 = Adj_STL[(train_length + 1):m]

S4 = svm(X12, Y12)

F_STL_SVR = as.numeric(predict(S4, X22)) + Sea_STL[(train_length + 1):m]
F_STL_SVR = ts(F_STL_SVR, frequency = 12, start = c(2017,1))

# STL + XGB
STL_train = Adj_STL[1:(train_length - 12)]
STL_test = Adj_STL[((train_length - 12) + 1):m]
n_steps = 12

# Prepare training set
data_train = prepare_data_xgb(STL_train, n_steps)
STL_X_train = data_train$X
STL_y_train = data_train$y

# Prepare testing set
data_test = prepare_data_xgb(STL_test, n_steps)
STL_X_test = data_test$X
STL_y_test = data_test$y

# === 2. Train XGBoost model ===

# Convert to xgbDMatrix
dtrain3 = xgb.DMatrix(data = STL_X_train, label = STL_y_train)
dtest3 = xgb.DMatrix(data = STL_X_test, label = STL_y_test)

# Train model
S5 = xgb.train(
  params = params,
  data = dtrain3,
  nrounds = 100,
  watchlist = list(train = dtrain3, test = dtest3),
  early_stopping_rounds = 10,
  verbose = 1
)

# === 3. Make Predictions ===
F_STL_XGB = predict(S5, dtest3) + Sea_STL[(train_length + 1):m]
F_STL_XGB = ts(F_STL_XGB, frequency = 12, start = c(2017,1))


# 6. STL + LSTM

# Define Model
model_3 = keras_model_sequential()
model_3$add(layer_lstm(units = 50, activation = "relu", input_shape = c(n_steps, n_features), return_sequences = FALSE))
model_3$add(layer_dense(units = 1))

# Compile Model
model_3$compile(
  optimizer = "adam",
  loss = "mse"
)

X_train = prepare_data(STL_train, n_steps)$X
X_train = array(as.numeric(unlist(X_train)), dim = c(length(X_train), n_steps, 1))

y_train = as.numeric(prepare_data(STL_train, n_steps)$y)

X_test = prepare_data(STL_test, n_steps)$X
X_test = array(as.numeric(unlist(X_test)), dim = c(length(X_test), n_steps, 1))

y_test = as.numeric(prepare_data(STL_test, n_steps)$y)

X_train_np = np_array(X_train)
y_train_np = np_array(y_train)

# Fit model
S6 = model_3$fit(
  X_train_np,
  y_train_np,
  batch_size = as.integer(12),
  epochs = as.integer(15),
  verbose = 2L
)


# Make predictions
F_STL_LSTM = model_3$predict(X_test) + Sea_STL[(train_length + 1):m]
F_STL_LSTM = ts(F_STL_LSTM, frequency = 12, start = c(2017,1))


# All Errors
errors = data.frame(
  Model = c("ARIMA", "ETS", "ANN", "SVR", "XGB", "LSTM",
            "X11-ARIMA", "X11-ETS", "X11-ANN", "X11-SVR", "X11-XGB", "X11-LSTM",
            "STL-ARIMA", "STL-ETS", "STL-ANN", "STL-SVR", "STL-XGB", "STL-LSTM"),
  
  MSE = c(
    mse(test1, F_ARIMA),
    mse(test1, F_ETS),
    mse(test1, F_ANN),
    mse(test1, F_SVR),
    mse(test1, F_XGB),
    mse(test1, F_LSTM),
    mse(test1, F_X11_ARIMA),
    mse(test1, F_X11_ETS),
    mse(test1, F_X11_ANN),
    mse(test1, F_X11_SVR),
    mse(test1, F_X11_XGB),
    mse(test1, F_X11_LSTM),
    mse(test1, F_STL_ARIMA),
    mse(test1, F_STL_ETS),
    mse(test1, F_STL_ANN),
    mse(test1, F_STL_SVR),
    mse(test1, F_STL_XGB),
    mse(test1, F_STL_LSTM)
  ),
  
  RMSE = c(
    rmse(test1, F_ARIMA),
    rmse(test1, F_ETS),
    rmse(test1, F_ANN),
    rmse(test1, F_SVR),
    rmse(test1, F_XGB),
    rmse(test1, F_LSTM),
    rmse(test1, F_X11_ARIMA),
    rmse(test1, F_X11_ETS),
    rmse(test1, F_X11_ANN),
    rmse(test1, F_X11_SVR),
    rmse(test1, F_X11_XGB),
    rmse(test1, F_X11_LSTM),
    rmse(test1, F_STL_ARIMA),
    rmse(test1, F_STL_ETS),
    rmse(test1, F_STL_ANN),
    rmse(test1, F_STL_SVR),
    rmse(test1, F_STL_XGB),
    rmse(test1, F_STL_LSTM)
  ),
  
  MASE = c(
    mase(test1, F_ARIMA),
    mase(test1, F_ETS),
    mase(test1, F_ANN),
    mase(test1, F_SVR),
    mase(test1, F_XGB),
    mase(test1, F_LSTM),
    mase(test1, F_X11_ARIMA),
    mase(test1, F_X11_ETS),
    mase(test1, F_X11_ANN),
    mase(test1, F_X11_SVR),
    mase(test1, F_X11_XGB),
    mase(test1, F_X11_LSTM),
    mase(test1, F_STL_ARIMA),
    mase(test1, F_STL_ETS),
    mase(test1, F_STL_ANN),
    mase(test1, F_STL_SVR),
    mase(test1, F_STL_XGB),
    mase(test1, F_STL_LSTM)
  ),
  
  MAE = c(
    mae(test1, F_ARIMA),
    mae(test1, F_ETS),
    mae(test1, F_ANN),
    mae(test1, F_SVR),
    mae(test1, F_XGB),
    mae(test1, F_LSTM),
    mae(test1, F_X11_ARIMA),
    mae(test1, F_X11_ETS),
    mae(test1, F_X11_ANN),
    mae(test1, F_X11_SVR),
    mae(test1, F_X11_XGB),
    mae(test1, F_X11_LSTM),
    mae(test1, F_STL_ARIMA),
    mae(test1, F_STL_ETS),
    mae(test1, F_STL_ANN),
    mae(test1, F_STL_SVR),
    mae(test1, F_STL_XGB),
    mae(test1, F_STL_LSTM)
  ),
  
  MAPE = c(
    mape(test1, F_ARIMA) * 100,
    mape(test1, F_ETS) * 100,
    mape(test1, F_ANN) * 100,
    mape(test1, F_SVR) * 100,
    mape(test1, F_XGB) * 100,
    mape(test1, F_LSTM) * 100,
    mape(test1, F_X11_ARIMA) * 100,
    mape(test1, F_X11_ETS) * 100,
    mape(test1, F_X11_ANN) * 100,
    mape(test1, F_X11_SVR) * 100,
    mape(test1, F_X11_XGB) * 100,
    mape(test1, F_X11_LSTM) * 100,
    mape(test1, F_STL_ARIMA) * 100,
    mape(test1, F_STL_ETS) * 100,
    mape(test1, F_STL_ANN) * 100,
    mape(test1, F_STL_SVR) * 100,
    mape(test1, F_STL_XGB) * 100,
    mape(test1, F_STL_LSTM) * 100
  )
)

print(errors)

# Round numeric columns to 4 decimal places
errors[, -1] = round(errors[, -1], 4)

# Print the updated table
print(errors)

# Save to Excel
write.xlsx(errors, "Mymensingh_Errors.xlsx")

# Forecast Plot
autoplot(ts1) +
  autolayer(F_ARIMA, series = "ARIMA") +
  autolayer(F_ETS, series ="ETS") +
  autolayer(F_ANN, series = "ANN") +
  autolayer(F_SVR, series = "SVR") +
  autolayer(F_XGB, series = "XGB") +
  autolayer(F_LSTM, series = "LSTM") +
  autolayer(F_X11_ARIMA, series = "X11-ARIMA") +
  autolayer(F_X11_ETS, series ="X11-ETS") +
  autolayer(F_X11_ANN, series = "X11-ANN") +
  autolayer(F_X11_SVR, series = "X11-SVR") +
  autolayer(F_X11_XGB, series = "X11-XGB") +
  autolayer(F_X11_LSTM, series = "X11-LSTM") +
  autolayer(F_STL_ARIMA, series = "STL-ARIMA") +
  autolayer(F_STL_ETS, series ="STL-ETS") +
  autolayer(F_STL_ANN, series = "STL-ANN") +
  autolayer(F_STL_SVR, series = "STL-SVR") +
  autolayer(F_STL_XGB, series = "STL-XGB") +
  autolayer(F_STL_LSTM, series = "STL-LSTM") +
  xlab("Year") + ylab("Thunderstorm Frequency") +
  theme_minimal() +
  theme(panel.border = element_rect(fill = NA, colour = "black", linewidth = 1),
        legend.title = element_blank(),
        legend.position = "bottom")


# Another Way

# Actual data
actual_df = data.frame(Time = time(ts1), Actual = as.numeric(ts1))

# Forecast data
forecast_df = data.frame(
  Time = as.numeric(time(F_ARIMA)),
  ARIMA = as.numeric(F_ARIMA),
  ETS = as.numeric(F_ETS),
  ANN = as.numeric(F_ANN),
  SVR = as.numeric(F_SVR),
  XGB = as.numeric(F_XGB),
  LSTM = as.numeric(F_LSTM),
  X11_ARIMA = as.numeric(F_X11_ARIMA),
  X11_ETS = as.numeric(F_X11_ETS),
  X11_ANN = as.numeric(F_X11_ANN),
  X11_SVR = as.numeric(F_X11_SVR),
  X11_XGB = as.numeric(F_X11_XGB),
  X11_LSTM = as.numeric(F_X11_LSTM),
  STL_ARIMA = as.numeric(F_STL_ARIMA),
  STL_ETS = as.numeric(F_STL_ETS),
  STL_ANN = as.numeric(F_STL_ANN),
  STL_SVR = as.numeric(F_STL_SVR),
  STL_XGB = as.numeric(F_STL_XGB),
  STL_LSTM = as.numeric(F_STL_LSTM)
)

forecast_long = forecast_df %>%
  pivot_longer(-Time, names_to = "Model", values_to = "Forecast")

ggplot() +
  geom_line(data = actual_df, aes(x = Time, y = Actual), color = "darkblue", linewidth = 0.6) + 
  geom_line(data = forecast_long, aes(x = Time, y = Forecast, color = Model), linewidth = 0.9) +
  labs(
    x = "Year",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
  )




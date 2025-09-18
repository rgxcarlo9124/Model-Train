#===============================IMPORT STATEMENTS AND THEIR PIP's==============================================
# Data fetching and manipulation
import yfinance as yf  # pip install yfinance

# Machine learning libraries
from sklearn.model_selection import train_test_split  # pip install scikit-learn
from catboost import CatBoostRegressor  # pip install catboost
from sklearn.tree import DecisionTreeRegressor  # pip install scikit-learn
from sklearn.ensemble import RandomForestRegressor  # pip install scikit-learn
from sklearn.ensemble import AdaBoostRegressor  # pip install scikit-learn
from prophet import Prophet  #pip install prophet


# Visualization/ploting
import plotly.graph_objs as go #pip install plotly


# Numerical data processing
import numpy as np  # Part of the Python standard library (no installation required)
import pandas as pd #pip install pandas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #pip install scikit-learn

#Date
from datetime import date

# Categorical data encoding
from sklearn.preprocessing import OrdinalEncoder  # pip install scikit-learn
from sklearn.preprocessing import MinMaxScaler  # pip install scikit-learn

# Deep learning library (for potential future use)
from tensorflow.keras.models import Sequential  # pip install tensorflow
from tensorflow.keras.layers import LSTM, Dense, Dropout  # pip install tensorflow

#------------------------USER INPUT PART----------------
Tic=input("Enter a Ticker Symbol ").upper()
Start_date=input("Enter Start Date in (YYYY-MM-DD) format ")
Date = input("Enter End Date in (YYYY-MM-DD) format or type (y) to put today's date: ")
if Date.lower() == "y":
    End_date = date.today().strftime('%Y-%m-%d')
else:
    End_date = Date
days=int(input('Enter number of days to forcast '))

#============================================CODING PART BEGINGS ================================================
# Download historical data
ticker_symbol = Tic
start_date = Start_date
end_date = End_date
Stock_data = yf.download(Tic, start=Start_date, end=End_date, progress=False)


#-------------Open--------------------------------------------------
Stock_data_open=Stock_data.copy()
Stock_data_open.drop(["High","Low","Close","Adj Close","Volume"],axis=1,inplace=True)
Stock_data_open.columns=["ds","y"]
m = Prophet(interval_width=0.95,daily_seasonality=True)
model = m.fit(Stock_data_open) #freq='D', epochs=100
future = m.make_future_dataframe(periods=days,freq="D")
forecast = m.predict(future)
New_close=forecast[["ds","yhat"]]
New_Stock_data=New_close.rename(columns={"ds":"Date",'yhat':'Open'})

#-------------High--------------------------------------------------
Stock_data_high=Stock_data.copy()
Stock_data_high.drop(["Open","Low","Close","Adj Close","Volume"],axis=1,inplace=True)
Stock_data_high.columns=["ds","y"]
m = Prophet(interval_width=0.95,daily_seasonality=True)
model = m.fit(Stock_data_high) #freq='D', epochs=100
future = m.make_future_dataframe(periods=days,freq="D")
forecast = m.predict(future)
New_high=forecast[["yhat"]]
New_Stock_data['High']=New_high

#---------------Low------------------------------------------------
Stock_data_low=Stock_data.copy()
Stock_data_low.drop(["Open","High","Close","Adj Close","Volume"],axis=1,inplace=True)
Stock_data_low.columns=["ds","y"]
m = Prophet(interval_width=0.95,daily_seasonality=True)
model = m.fit(Stock_data_low) #freq='D', epochs=100
future = m.make_future_dataframe(periods=days,freq="D")
forecast = m.predict(future)
New_low=forecast[["yhat"]]
New_Stock_data['Low']=New_low

#---------------Close------------------------------------------------
Stock_data_close=Stock_data.copy()
Stock_data_close.drop(["Open","High","Low","Adj Close","Volume"],axis=1,inplace=True)
Stock_data_close.columns=["ds","y"]
m = Prophet(interval_width=0.95,daily_seasonality=True)
model = m.fit(Stock_data_close) #freq='D', epochs=100
future = m.make_future_dataframe(periods=days,freq="D")
forecast = m.predict(future)
New_close=forecast[["yhat"]]
New_Stock_data['Close']=New_close

#---------------Adj_Close------------------------------------------------
Stock_data_Adj_Close=Stock_data.copy()
Stock_data_Adj_Close.drop(["Open","High","Low","Close","Volume"],axis=1,inplace=True)
Stock_data_Adj_Close.columns=["ds","y"]
m = Prophet(interval_width=0.95,daily_seasonality=True)
model = m.fit(Stock_data_Adj_Close) #freq='D', epochs=100
future = m.make_future_dataframe(periods=days,freq="D")
forecast = m.predict(future)
New_Adj_close=forecast[["yhat"]]
New_Stock_data['Adj_Close']=New_Adj_close


#-------------Volume------------------------------------------------
Stock_data_Volume=Stock_data.copy()
Stock_data_Volume.drop(["Open","High","Low","Close","Adj Close"],axis=1,inplace=True)
Stock_data_Volume.columns=["ds","y"]
m = Prophet(interval_width=0.95,daily_seasonality=True)
model = m.fit(Stock_data_Volume) #freq='D', epochs=100
future = m.make_future_dataframe(periods=days,freq="D")
forecast = m.predict(future)
New_Volume=forecast[["yhat"]]
New_Stock_data['Volume']=New_Volume


#----------------UPDATION OF New Stock data--------------------
Stock_data.set_index('Date', inplace=True)
New_Stock_data.set_index('Date', inplace=True)

# Update df2 with df1 values based on the date index
New_Stock_data.update(Stock_data)

# Reset index to make 'date' a column again
New_Stock_data.reset_index(inplace=True)
#-------------------------------------------------------------


# Feature Engineering
New_Stock_data['DayOfWeek'] = New_Stock_data['Date'].dt.dayofweek
New_Stock_data['Month'] = New_Stock_data['Date'].dt.month
New_Stock_data['DayOfMonth'] = New_Stock_data['Date'].dt.day

New_Stock_data = New_Stock_data.dropna()

# Create feature matrix X and target variable y
target_col = "Close"

X = New_Stock_data.drop([target_col], axis=1)
y = New_Stock_data[target_col]

# Create an OrdinalEncoder for datetime features
ordinal_encoder = OrdinalEncoder()
X_encoded = ordinal_encoder.fit_transform(X)

# Split the encoded data into training and testing sets for all models
X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# CATBOOST REGRESSOR
catboost_model = CatBoostRegressor(iterations=100, task_type='GPU', depth=6, learning_rate=0.1, loss_function='RMSE', random_seed=42)
catboost_model.fit(X_train, y_train)

# Create decision tree model
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train_encoded, y_train)

# Create Random Forest model
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train_encoded, y_train)

#Create Adaboost Regressor
ada_boost_regressor = AdaBoostRegressor(n_estimators=100, random_state=42)
ada_boost_regressor.fit(X_train_encoded, y_train)

# MAKING PREDICTIONS
catboost_predictions = catboost_model.predict(New_Stock_data)
predictions_dt = decision_tree_model.predict(ordinal_encoder.transform(New_Stock_data.drop([target_col], axis=1)))
predictions_rf = random_forest_model.predict(ordinal_encoder.transform(New_Stock_data.drop([target_col], axis=1)))
predictions_ab = ada_boost_regressor.predict(ordinal_encoder.transform(New_Stock_data.drop([target_col], axis=1)))
#=======================================================================================================================




#-------------------------------------------------------LSTM-----------------------------------------------------------------

# Data Preprocessing for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(New_Stock_data[target_col].values.reshape(-1, 1))

# Define window size for LSTM
window_size = 60

# Create sequences of data with window size
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(scaled_data, window_size)

# Split the data into training and testing sets for LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Define LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

# Evaluate the model
loss = model.evaluate(X_test_lstm, y_test_lstm)
print(f"LSTM Model Loss: {loss:.6f}")

# Convert Stock_data to a numpy array
Stock_data_array = New_Stock_data[target_col].values.reshape(-1, 1)

# Scale the input data
scaled_stock_data = scaler.transform(Stock_data_array)

# Create sequences for prediction
X_pred, _ = create_sequences(scaled_stock_data, window_size)

# Reshape the data to match the input shape of the LSTM model
X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

# Make predictions
predictions_lstm = model.predict(X_pred)

# Inverse transform the predictions to get the actual stock prices
predictions_lstm = scaler.inverse_transform(predictions_lstm)

#----------------------------------------------------------------------------------------
# Convert predictions_lstm to a flat list or Series
predictions_lstm_flat = [item[0] for item in predictions_lstm]

# Convert the list to a DataFrame or Series
predictions_df = pd.Series(predictions_lstm_flat, name='LSTM_Predicted_Close')

# Ensure the length of predictions matches the corresponding dates in New_Stock_data
predictions_df = predictions_df[-len(predictions_lstm):].reset_index(drop=True)

# Add the predictions as a new column to New_Stock_data
New_Stock_data['LSTM_Predicted_Close'] = predictions_df


#--------------------------------------------------------------------------------------------------------------------------
# F1 score and accuracy are not applicable for regression tasks.
# If you want to evaluate a classification task, you need to define your target variable accordingly.

# The F1 score is a classification metric and is not applicable when  target variable is continuous (as in regression).
# Set a threshold to convert predictions to binary classes (1 if above threshold, 0 otherwise)



def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r_squared = r2_score(y_true, y_pred)
    return mae, mse, rmse, mape, r_squared

def sortino_ratio(returns, rf=0):
    downside_returns = np.minimum(0, returns - rf)
    expected_return = np.mean(returns - rf)
    downside_std = np.std(downside_returns)
    sortino_ratio = expected_return / downside_std if downside_std != 0 else np.inf
    return sortino_ratio

def jensens_alpha(returns, benchmark_returns, rf=0):
    excess_returns = returns - rf
    excess_benchmark_returns = benchmark_returns - rf
    beta = np.cov(excess_returns, excess_benchmark_returns)[0, 1] / np.var(excess_benchmark_returns)
    alpha = np.mean(excess_returns) - beta * np.mean(excess_benchmark_returns)
    return alpha

def max_drawdown(prices):
    prices = np.array(prices)  # Ensure prices is a numpy array
    cumulative_return = (prices / prices[0]) - 1
    running_max = np.maximum.accumulate(cumulative_return)
    drawdown = running_max - cumulative_return
    max_drawdown = np.max(drawdown)
    return max_drawdown

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))

def mean_directional_accuracy(y_true, y_pred):
    return np.mean((y_pred[1:] - y_true[:-1]) * (y_true[1:] - y_true[:-1]) > 0)

# Calculate metrics for all models
def evaluate_models(y_true, y_pred, model_name):
    mae, mse, rmse, mape, r_squared = calculate_metrics(y_true, y_pred)
    returns = np.diff(y_true) / y_true[:-1]
    benchmark_returns = np.diff(y_pred) / y_pred[:-1]
    sortino = sortino_ratio(returns)
    alpha = jensens_alpha(returns, benchmark_returns)
    mdd = max_drawdown(y_true)
    da = directional_accuracy(y_true, y_pred)
    mda = mean_directional_accuracy(y_true, y_pred)
    print(f"     Evaluation for  {model_name}:")
    print("----------------------------------------------------------------")
    print(f"Mean Absolute Error (MAE)                : {mae}")
    print(f"Mean Squared Error (MSE)                 : {mse}")
    print(f"Root Mean Squared Error (RMSE)           : {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE)    : {mape}")
    print(f"R-squared (R²)                           : {r_squared}")
    print(f"Sortino Ratio (Sortino)                  : {sortino}")
    print(f"Jensen's Alpha (Alpha)                   : {alpha}")
    print(f"Maximum Drawdown (MDD)                   : {mdd}")
    print(f"Directional Accuracy (DA)                : {da}")
    print(f"Mean Directional Accuracy (MDA)          : {mda}")
    print("=================================================================")

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R_squared": r_squared,
        "Sortino": sortino,
        "Alpha": alpha,
        "MDD": mdd,
        "DA": da,
        "MDA": mda
    }

# List to store evaluation results
results = []


# Evaluate CatBoost
results.append(evaluate_models(y_test, catboost_model.predict(X_test), "CatBoost"))

# Evaluate Decision Tree
results.append(evaluate_models(y_test, decision_tree_model.predict(X_test_encoded), "Decision Tree"))

# Evaluate Random Forest
results.append(evaluate_models(y_test, random_forest_model.predict(X_test_encoded), "Random Forest"))

# Evaluate AdaBoost
results.append(evaluate_models(y_test, ada_boost_regressor.predict(X_test_encoded), "AdaBoost"))

# Evaluate LSTM
# Ensure LSTM predictions are aligned with the test set
y_test_lstm_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
results.append(evaluate_models(y_test_lstm_actual, predictions_lstm[-len(y_test_lstm_actual):].flatten(), "LSTM"))



# Dictionary to map metric abbreviations to their full names
metric_full_names = {
    "MAE": "Mean Absolute Error",
    "MSE": "Mean Squared Error",
    "RMSE": "Root Mean Squared Error",
    "MAPE": "Mean Absolute Percentage Error",
    "R_squared": "R-squared",
    "Sortino": "Sortino Ratio",
    "Alpha": "Jensen's Alpha",
    "MDD": "Maximum Drawdown",
    "DA": "Directional Accuracy",
    "MDA": "Mean Directional Accuracy"
}

# Compare models based on metrics
best_models = {}
for metric in metric_full_names.keys():
    if metric in ["MAE", "MSE", "RMSE", "MAPE", "MDD"]:
        best_models[metric] = min(results, key=lambda x: x[metric])["Model"]
    else:
        best_models[metric] = max(results, key=lambda x: x[metric])["Model"]

# Print best models for each metric with full names
print("-=-=-=-=-=-=-=-=-=Best models for each metric=-=-=-=-=-=-=-=-=-=")
for metric, model in best_models.items():
    print(f"       {metric_full_names[metric]}:{model}")




# Plotting
fig = go.Figure()

fig.add_trace(go.Scatter(x=New_Stock_data["Date"], y=predictions_dt, mode='lines', name='Decision Tree', line=dict(color='green')))
fig.add_trace(go.Scatter(x=New_Stock_data["Date"], y=catboost_predictions, mode='lines', name='CatBoost', line=dict(color='red')))
fig.add_trace(go.Scatter(x=New_Stock_data["Date"][-len(predictions_lstm):], y=New_Stock_data['LSTM_Predicted_Close'], mode='lines', name='LSTM', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=New_Stock_data["Date"], y=predictions_rf, mode='lines', name='Random Forest', line=dict(color='purple')))
fig.add_trace(go.Scatter(x=New_Stock_data["Date"], y=predictions_ab, mode='lines', name='Ada Boost', line=dict(color='magenta')))
fig.add_trace(go.Scatter(x=New_Stock_data["Date"], y=Stock_data["Close"], mode='lines', name='Actual Close', line=dict(color='blue')))

# Update layout with hovermode for trackball and crosshair
fig.update_layout(
    title={
        'text': f'STOCK MARKET FORECAST OF {ticker_symbol} USING SOME MODELS',
        'x': 0.5,
        'xanchor': 'center',
        'font': {
            'size': 20
        }
    },
    xaxis_title={
        'text': 'DATE',
        'font': {
            'size': 18
        }
    },
    yaxis_title={
        'text': 'CLOSE PRICE',
        'font': {
            'size': 18
        }
    },

    hovermode='x unified',# Enable trackball and crosshair,
    xaxis=dict(showspikes=True),
    yaxis=dict(showspikes=True),
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Tahoma"
    ),
    legend=dict(x=0, y=1, traceorder='normal', font=dict(size=19)),font_family="Tahoma",
    margin=dict(l=50, r=50, t=50, b=50),  # Center the graph by setting equal margins
    height=940,
    width=1860,
)

# Show the figure
fig.show()




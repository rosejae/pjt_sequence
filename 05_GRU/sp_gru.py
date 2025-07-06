import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import SGD  

#
# load data
#

AMZN = yf.download('AMZN', 
                    start='2014-01-01', 
                    end='2019-12-31', 
                    progress=False,
                    )

all_data = AMZN[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)

#
# preprocessing
#

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(all_data[['Open']])  # 'Open'만 정규화

def ts_train_test(all_data, time_steps, for_periods): 
    ts_train = all_data[:'2018'].iloc[:, 0:1].values
    ts_test  = all_data['2019':].iloc[:, 0:1].values
    ts_train_len = len(ts_train) # 1258
    ts_test_len = len(ts_test) # 251

    X_train = []
    y_train = []
    y_train_stacked = []
    for i in range(time_steps, ts_train_len-1): 
        X_train.append(ts_train[i-time_steps:i, 0]) 
        y_train.append(ts_train[i:i+for_periods, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    inputs = pd.concat((all_data["Open"][:'2018'], all_data["Open"]['2019':]), axis=0).values
    inputs = inputs[len(inputs)-len(ts_test)-time_steps:]
    inputs = inputs.reshape(-1, 1) 

    X_test = []
    for i in range(time_steps, ts_test_len+time_steps-for_periods):
        X_test.append(inputs[i-time_steps:i, 0]) 
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test

time_steps = 5
for_periods = 2

X_train, y_train, X_test = ts_train_test(all_data, time_steps, for_periods)

#
# train
#

model = Sequential()
# model.add(Input(shape=(time_steps, 1))) 
# # inputs = Input() 이런식으로 inputs이 다음 레이어에 들어갈때 Input 레이어를 사용하는 듯 
model.add(GRU(units=40, return_sequences=True, activation='tanh', input_shape=(time_steps, 1)))
model.add(GRU(units=30, activation='tanh'))
model.add(Dense(units=2))

model.compile(optimizer=SGD(
    learning_rate=0.01, 
    decay=1e-7, 
    momentum=0.9, 
    nesterov=False), loss='mean_squared_error')
model.summary()
model.fit(X_train, y_train, epochs=20, batch_size=60, verbose=1)

#
# inference
#

predictions_scaled = model.predict(X_test)
# predictions_original = scaler.inverse_transform(predictions_scaled)
print(predictions_scaled[:5])
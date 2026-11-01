import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

dates=pd.date_range('2018-01-01',periods=1500,freq='D')
signal=0.02*np.arange(len(dates))+np.sin(2*np.pi*dates.dayofyear/365.25)
noise=0.3*np.random.randn(len(dates))
values=signal+noise
df=pd.DataFrame({'date':dates,'value':values})

plt.figure(figsize=(10,4))
plt.plot(df['date'],df['value'])
plt.title('Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

scaler=MinMaxScaler()
v=scaler.fit_transform(df[['value']])

INPUT_LEN=120
HORIZON=1

def windows(data,n,h):
    X,y=[],[]
    for i in range(len(data)-n-h+1):
        X.append(data[i:i+n])
        y.append(data[i+n:i+n+h])
    return np.array(X),np.array(y)

X,y=windows(v,INPUT_LEN,HORIZON)

test_size=300
X_train,X_test=X[:-test_size],X[-test_size:]
y_train,y_test=y[:-test_size],y[-test_size:]

def build_lstm(input_len=INPUT_LEN,horizon=HORIZON,units=64,dropout=0.1):
    m=keras.Sequential([
        layers.Input(shape=(input_len,1)),
        layers.LSTM(units,return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(units//2),
        layers.Dropout(dropout),
        layers.Dense(horizon)])
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=[keras.metrics.MeanAbsoluteError(name='mae')])
    return m

m=build_lstm()
m.summary()

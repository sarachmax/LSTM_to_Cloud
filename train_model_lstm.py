# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 11:14:13 2018

@author: sarac
"""

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

print("Open CSV File ... ")
dataset = pd.read_csv("XAUUSD_1M_selected.csv")
dataset.Date = pd.to_datetime(dataset.Date)

print("Preprocessing Data ... ")
start_date= dataset.Date >= pd.Timestamp("2016-01-03 00:00:00")
end_date= dataset.Date < pd.Timestamp("2018-01-01 00:00:00")
dataset = dataset.loc[start_date & end_date]
dataset = dataset.iloc[:,3:6].values

sc = MinMaxScaler(feature_range = (1,2))
num_obs = 512
num_pred = 16 

input_data = [] 
output_data = [] 
print("Scale input and output before Training")
for i in range(num_obs, len(dataset)-num_pred):
    data_windows = dataset[i-num_obs:i+num_pred,:]
    data_windows = sc.fit_transform(data_windows)
    input_data.append(data_windows[:num_obs,:])
    output_data.append(data_windows[num_obs:num_obs+num_pred,2])
    
input_data, output_data = np.array(input_data), np.array(output_data)
input_shape=(input_data.shape[1], input_data.shape[2])

print("Clear unused variables")
data_windows=None
dataset=None

print("====== Start Training ======")    

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, TimeDistributed
from keras.callbacks import  EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

"""
regressor.add(TimeDistributed(Conv1D(nb_filter=256, filter_length=16, input_shape=input_shape, activation="relu")))
regressor.add(TimeDistributed(MaxPooling1D(pool_size=8)))
regressor.add(TimeDistributed(Dropout(0.2)))
regressor.add(TimeDistributed(Flatten()))

regressor.add(Conv1D(nb_filter=256, filter_length=16, input_shape=input_shape, activation="relu"))
regressor.add(MaxPooling1D(pool_size=8))
regressor.add(Dropout(0.2))

# regressor.add(LSTM(units = 256, return_sequences = True, input_shape =(input_data.shape[1], input_data.shape[2])))

regressor.add(LSTM(units = 256, return_sequences = True))
regressor.add(Dropout(0.2))  

regressor.add(LSTM(units = 128, return_sequences = True))
regressor.add(Dropout(0.2)) 

regressor.add(LSTM(units = 64))
regressor.add(Dropout(0.2))
"""
regressor = Sequential()

regressor.add(LSTM(units = 256, return_sequences = True, input_shape=input_shape, activation="relu"))
regressor.add(Dropout(0.25))  

regressor.add(LSTM(units = 128, return_sequences = True, activation="relu"))
regressor.add(Dropout(0.25)) 

regressor.add(LSTM(units = 64, return_sequences = True, activation="relu"))
regressor.add(Dropout(0.25)) 

regressor.add(LSTM(units = 32, activation="relu"))
regressor.add(Dropout(0.25))

regressor.add(Dense(units = 16, activation='relu'))     
regressor.summary()

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

mcp = ModelCheckpoint('lstm_weights/weights{epoch:04d}.h5', save_weights_only=True, period=5)
tb = TensorBoard('logs')
#es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, verbose=1)

regressor.fit(input_data, output_data, epochs = 300, shuffle=True,
              callbacks=[rlr,mcp, tb], validation_split=0.2, verbose=1, batch_size=64)

model_json = regressor.to_json()
with open('lstm_weights/last_model.json', 'w') as json_file:
    json_file.write(model_json)
regressor.save_weights('lstm_weights/last_model.h5')
print('saved model')

print(" === save weight done ===")






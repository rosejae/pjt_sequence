import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Dense, 
    Flatten, 
    Dropout, 
    LSTM, 
    TimeDistributed, 
    Conv1D, 
    MaxPooling1D,
    )

#
# preprocessing
#

def load_file(filepath):  
	dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)  
        loaded.append(data) 
    loaded = np.dstack(loaded) 
    return loaded 

def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'  
    
    filenames = list()
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    
    X = load_group(filenames, filepath)
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

def load_dataset(prefix=''):
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    trainy = trainy - 1  
    testy = testy - 1
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    return trainX, trainy, testX, testy

trainX, trainy, testX, testy = load_dataset()

n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
n_steps, n_length = 4, 32
trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))

#
# train
#

model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), 
                          input_shape=(None, n_length, n_features)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

save_path = 'stack_model.weights.h5'
if os.path.exists(save_path):
    model.load_weights(save_path) # 추론용으로 사용
else:
    model.fit(trainX, trainy, epochs=25, batch_size=32, verbose=1)
    model.save_weights(save_path)

#
# inference
#

_, accuracy = model.evaluate(testX, testy, batch_size=32, verbose=0)
print(accuracy)
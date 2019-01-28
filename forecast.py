from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from dateparser import parse
import datetime

def get_model(weights):  # function to load LSTM model and scaling model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.load_weights('weights/'+weights+'.hdf5')
    scale = joblib.load('scale_models/scale_'+weights+'.pkl')
    return model, scale

def get_forecast(seed, data='google'): #seed = date string or previous day closing price, data = 'google' or 'msft' which model to use
    model, scale = get_model(data) #load model based on data  
    if type(seed) == float or type(seed) == int:
        x = scale.transform(float(seed))
        pred = model.predict(np.expand_dims(x, axis=0))
        return scale.inverse_transform(pred)
    elif type(seed) == str:
        date = parse(seed)
        today = parse('30-Jun-17') if data=='google' else parse('2018-06-20') 
        if date > today:
            steps = (date - today).days
            seed = 929.68 if data == 'google' else 101.87 #last closing price of respective dataset to use as seed for predicting closing price of future dates
            seed = scale.transform(seed)
            for i in range(1,steps+1):
                seed = np.array(seed,ndmin=3)
                seed = model.predict(seed).tolist()[0][0]
            return scale.inverse_transform(seed)
        else:
            print "Invalid Date"
    else:
        print "Please provide input parameters correctly"
        
if __name__ == '__main__':
    pred = get_forecast(62,'msft')
    print "Next closing value is",pred
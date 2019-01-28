from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import dateparser
import numpy as np

np.random.seed(100) #for reproducibility

def read_data():
    data = pd.read_csv('./dataset/tbl_aal.csv', header=None, thousands=",")
    data.drop(data.columns[0], axis=1,inplace=True)
    data.dropna(inplace=True)
    data.columns = ['Symbol','Date','Open','High','Low','Close','Adj.Close','Volume']
    high = np.array([float(i.replace(',','')) for i in data['High'][::-1] if (i != '-' and type(i) != float)])
    low = np.array([float(i.replace(',','')) for i in data['Low'][::-1] if (i != '-' and type(i) != float)])    
    return high, low #reversing the stock prices from previous date to till date

def create_XY(data): #converting the univariate time series into multivariate time series(making it supervised learning)
    X_high = data['high']
    Y_high = X_high[1:]
    X_high = X_high[0:-1]
    X_low = data['low']
    Y_low = X_low[1:]
    X_low = X_low[0:-1]
    X = np.array([X_high, X_low]).transpose()
    Y = np.array([Y_high, Y_low]).transpose()
    return X,Y

def scale_data(data): #scaling data between 0 and 1 because LSTMs are sensitive to the scale of the input data.
    scale = MinMaxScaler(feature_range=(0, 1))
    high_scale = scale.fit(np.array(data['high']).reshape(-1,1))
    low_scale = scale.fit(np.array(data['low']).reshape(-1,1))
    data['high'] = high_scale.transform(np.array(data['high']).reshape(-1, 1))
    data['low'] = low_scale.transform(np.array(data['low']).reshape(-1, 1))
    return high_scale, low_scale, data

def get_train_test(train_size=0.9): #dividing the dataset into train and test set. Default train size is 90%
    high, low = read_data()
    tmp = pd.DataFrame({'high':high, 'low':low})
    train, test = tmp.iloc[0:int(train_size*len(tmp))], tmp.iloc[int(train_size*len(tmp)):]
    print(train.shape)
    print(test.shape)
    high_scale, low_scale, train = scale_data(train) #fir the scaler model on train data
    print(train.shape)
    test['high'] = high_scale.transform(np.array(test['high']).reshape(-1,1)) #scaling test set
    test['low'] = low_scale.transform(np.array(test['low']).reshape(-1,1))
    x_train, y_train = create_XY(train) # creating supervised train data
    x_test, y_test = create_XY(test) # creating supervised test data
    return x_train, y_train, x_test, y_test, high_scale, low_scale

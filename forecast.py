import pickle
import numpy as np
import pandas as pd

with open('model_scalars.pkl','rb') as f:  # Python 3: open(..., 'rb')
    high_scale,low_scale,model = pickle.load(f)


def get_forecast(seed_h,seed_l): #seed = date string or previous day closing price, data = 'google' or 'msft' which model to use
    x_h = high_scale.transform(np.array(float(seed_h)).reshape(-1,1))
    x_l = low_scale.transform(np.array(float(seed_l)).reshape(-1,1))
    pred = np.array([x_h,x_l])
    print(pred.shape)
    return high_scale.inverse_transform(pred[0]), low_scale.inverse_transform(pred[1])
        
if __name__ == '__main__':
    ip_high = input("Please enter the last value of High: ")
    ip_low = input("Please enter the last value of Low: ")
    pred = get_forecast(ip_high,ip_low)
    print("Next forecasted High value is",pred[0])
    print("Next forecasted Low value is",pred[1])
    
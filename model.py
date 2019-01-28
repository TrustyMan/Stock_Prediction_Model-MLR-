from sklearn.metrics import mean_squared_error
# from matplotlib import pyplot
from utils import get_train_test
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import warnings
import matplotlib.pyplot as plt
import pickle

warnings.filterwarnings('ignore')

model = MultiOutputRegressor(LinearRegression(),n_jobs=-1)

x_train, y_train, x_test, y_test, high_scale, low_scale = get_train_test() # train test split of dataset

# model = LinearRegression()
model.fit(x_train, y_train)
model.predict()

with open('model_scalars.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([high_scale,low_scale,model], f)

test = model.predict(x_test)
y_trueh = high_scale.inverse_transform(y_test[:,0].reshape(-1,1)).ravel() # inverse scaling of the original test data. inverse scaling is done to get the original stock value which was converted into 0 and 1 range
y_predh = high_scale.inverse_transform(test[:,0].reshape(-1,1)).ravel() # inverse scaling of the predicted data
y_truel = low_scale.inverse_transform(y_test[:,1].reshape(-1,1)).ravel() # inverse scaling of the original test data. inverse scaling is done to get the original stock value which was converted into 0 and 1 range
y_predl = low_scale.inverse_transform(test[:,1].reshape(-1,1)).ravel()
plot = pd.DataFrame(np.stack([y_trueh,y_predh],axis=1), columns=["True","Predicted"]) #creating dataframe of true and predicted values for ploting
plot.plot()
plot = pd.DataFrame(np.stack([y_truel,y_predl],axis=1), columns=["True","Predicted"]) #creating dataframe of true and predicted values for ploting
plot.plot()
plt.show()
err = np.sqrt(mean_squared_error(y_trueh, y_predh)) #calculation of root mean square error
print("Root Mean Squared Error of High is :%s"%err)
err = np.sqrt(mean_squared_error(y_truel, y_predl)) #calculation of root mean square error
print("Root Mean Squared Error of Low is :%s"%err)
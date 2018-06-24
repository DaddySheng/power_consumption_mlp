# _*_ coding: utf-8 _*_
"""
INFORMATION:
@Author: Leo Sheng
@File: PyCharm  Power-Consumption-Prediction-master  test_1.py
@Time: 2018/06/24 17:02
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

#create dataset
df = pd.read_csv('D:/ML/Tianchi_power.csv')
df['record_date'] = pd.to_datetime(df['record_date'])
s_power_consumption = df.groupby('record_date')['power_consumption'].sum()
s_power_consumption.index = pd.to_datetime(s_power_consumption.index).sort_values()
day_type = [3, 3, 3, 6, 7, 3, 3]
rest_days = []
if s_power_consumption.size % 7 == 0:
    num_weeks = s_power_consumption.size / 7
else:
    num_rest_days = s_power_consumption.size % 7
    rest_days = day_type[0:num_rest_days]

s_day_type = pd.Series(data=day_type * num_weeks + rest_days, index=s_power_consumption.index)

#data proprecessing
std_sca = StandardScaler().fit(s_power_consumption.values.reshape(-1,1))
data_std = StandardScaler().fit_transform(s_power_consumption.values.reshape(-1,1)).flatten()

prediction_period = 30
input_size = 120
reg = MLPRegressor(activation = 'relu',hidden_layer_sizes = (300,),
                               max_iter=8000,verbose=True,learning_rate='adaptive',
                               tol=0.0,warm_start=True,solver='adam')
window_size = input_size + prediction_period
seq_length = data_std.size

X_power = []
XY_day_type = []
Y_power = []

for i in xrange(0, seq_length - window_size):
    xy_power = data_std[i:window_size + i]
    x_power = xy_power[0:input_size]
    X_power.append(x_power)
    y_power = xy_power[-prediction_period:]
    Y_power.append(y_power)

    xy_day_type = s_day_type.values[i:window_size + i]
    XY_day_type.append(xy_day_type)

X_power = np.array(X_power)
XY_day_type = np.array(XY_day_type)
X = np.concatenate((X_power,XY_day_type),axis = 1)
enc = OneHotEncoder(categorical_features=np.arange(window_size-prediction_period,X.shape[1]))
X = enc.fit_transform(X)
Y = np.array(Y_power)

# the last month for testing
X = X.toarray()
X_train = X[:-1]; X_test = X[-1]
Y_train = Y[:-1]; Y_test = Y[-1]
reg.fit(X_train,Y_train)
X_test = X_test.reshape(1,-1)
pred_y = reg.predict(X_test)

pred = std_sca.inverse_transform(pred_y.reshape(-1,1))
test = std_sca.inverse_transform(Y_test.reshape(-1,1))
err = abs(pred-test)/test

plt.plot(pred.flatten(),label='predict')
plt.plot(test.flatten(),label='real')
plt.ylim((0,5500000))
plt.legend()
plt.show()

plt.plot(err,label='err')
plt.legend()
plt.show()

# 误差方差
re_err = abs(pred-test)
mean_fit_err = abs(reg.predict(X_train)-Y_train).sum().mean()
mean_pre_err = re_err.mean()
print 'fit err:', mean_fit_err
print 'pre err', mean_pre_err

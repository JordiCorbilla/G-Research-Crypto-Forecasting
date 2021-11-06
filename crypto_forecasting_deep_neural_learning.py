# -*- coding: utf-8 -*-
# Copyright 2021 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pandas as pd
import os
import secrets
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from stock_prediction_plotter import Plotter
import numpy as np

TIME_STEPS = 3

pd.options.mode.chained_assignment = None

# Start with a small subset of the data (5000 rows)
fields = ['timestamp', 'Asset_ID', 'Close']
train_df = pd.read_csv('C:/Users/thund/Downloads/g-research-crypto-forecasting/train.csv', low_memory=False, 
                       dtype={'Asset_ID': 'int8', 'Count': 'int32', 'row_id': 'int32', 'Count': 'int32', 
                              'Open': 'float64', 'High': 'float64', 'Low': 'float64', 'Close': 'float64', 
                              'Volume': 'float64', 'VWAP': 'float64'
                             }
                      , nrows=5000
                      , usecols=fields)
train_df.dropna(axis = 0, inplace = True)
print(train_df.head(3))

# Print the list of Assets
print(train_df['Asset_ID'].unique())

# Let's try BTC
train_btc = train_df[train_df['Asset_ID'] == 1]
del train_btc['Asset_ID']
print(train_btc.head(3))

# 70% of the data for training, the rest for validation
training_data = train_btc[:int(train_btc.shape[0]*0.7)].copy()
test_data = train_btc[int(train_btc.shape[0]*0.7):].copy()
print(training_data.head(3))
print(test_data.head(3))

training_data = training_data.set_index('timestamp')
test_data = test_data.set_index('timestamp')

min_max = MinMaxScaler(feature_range=(0, 1))
train_scaled = min_max.fit_transform(training_data)

print('mean:', train_scaled.mean(axis=0))
print('max', train_scaled.max())
print('min', train_scaled.min())
print('Std dev:', train_scaled.std(axis=0))

# Training Data Transformation
x_train = []
y_train = []
for i in range(TIME_STEPS, train_scaled.shape[0]):
    x_train.append(train_scaled[i - TIME_STEPS:i])
    y_train.append(train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

total_data = pd.concat((training_data, test_data), axis=0)
inputs = total_data[len(total_data) - len(test_data) - TIME_STEPS:]
test_scaled = min_max.fit_transform(inputs)

# Testing Data Transformation
x_test = []
y_test = []
for i in range(TIME_STEPS, test_scaled.shape[0]):
    x_test.append(test_scaled[i - TIME_STEPS:i])
    y_test.append(test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print(x_train)
print(y_train)

STOCK_TICKER = 'BTC'
TODAY_RUN = datetime.today().strftime("%Y%m%d")
TOKEN = STOCK_TICKER + '_' + TODAY_RUN + '_' + secrets.token_hex(16)
PROJECT_FOLDER = os.path.join(os.getcwd(), TOKEN)

if not os.path.exists(PROJECT_FOLDER):
    os.makedirs(PROJECT_FOLDER)
        
plotter = Plotter(True, PROJECT_FOLDER, STOCK_TICKER, STOCK_TICKER, STOCK_TICKER)
plotter.plot_histogram_data_split(training_data, test_data, 70)

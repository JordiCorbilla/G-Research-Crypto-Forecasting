# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 16:23:26 2021

@author: jordicorbilla
"""

import pandas as pd

pd.options.mode.chained_assignment = None

# Start with a small subset of the data (5000 rows)
train_df = pd.read_csv('C:/Users/thund/Downloads/g-research-crypto-forecasting/train.csv', low_memory=False, 
                       dtype={'Asset_ID': 'int8', 'Count': 'int32', 'row_id': 'int32', 'Count': 'int32', 
                              'Open': 'float64', 'High': 'float64', 'Low': 'float64', 'Close': 'float64', 
                              'Volume': 'float64', 'VWAP': 'float64'
                             }
                      , nrows=5000)
train_df.dropna(axis = 0, inplace = True)
print(train_df.head(3))

# Print the list of Assets
print(train_df['Asset_ID'].unique())

# Let's try BTC
train_btc = train_df[train_df['Asset_ID'] == 1]
print(train_btc.head(3))


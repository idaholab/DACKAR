import numpy as np
import pandas as pd
import os
import sys

cwd = os.getcwd()

frameworkDir = os.path.abspath(os.path.join(cwd, os.pardir, 'src'))
sys.path.append(frameworkDir)

data_path = os.path.abspath(os.path.join(cwd, os.pardir, 'data'))
taxi_data_file = os.path.abspath(os.path.join(data_path, 'nyc_taxi_passengers.csv'))

from dackar.anomalies.MatrixProfile import MatrixProfile


taxi_df = pd.read_csv(taxi_data_file, index_col='timestamp')
taxi_df['value'] = taxi_df['value'].astype(np.float64)
taxi_df.index = pd.to_datetime(taxi_df.index, format='%m/%d/%y %H:%M')
taxi_df.head()

steam_gen_data_file = os.path.abspath(os.path.join(data_path, 'Steamgen.csv'))
steam_df = pd.read_csv(steam_gen_data_file)
steam_df.head()

m = 48
mpObj = MatrixProfile(m, normalize='robust', method='incremental')
mpObj.fit(taxi_df.iloc[0:1000])
fig = mpObj.plot()

mpObj.evaluate(taxi_df.iloc[1000:])
fig = mpObj.plot()

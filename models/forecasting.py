import numpy as np
import pandas as pd
import sktime
import pytimetk as tk
from pytimetk import summarize_by_time
from pytimetk import anomalize
import plotly
import os

from data_manipulation.data_wrangling import summary_by_time,air_category_summary_by_time, summary_by_time_weekly, anamoly_detection,data_for_classification
from datetime import datetime, timedelta
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.dists_kernels import FlatDist, ScipyDist
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

analysis_data = pd.read_csv('HistoricalObs.csv', 
                   encoding = 'unicode_escape')

analysis_data['Date'] = pd.to_datetime(analysis_data['Date'])

# key_data = analysis_data[analysis_data['Parameter_ParameterDescription'] != 'Temperature']

key_data = analysis_data[analysis_data['Parameter_ParameterDescription'] == 'Temperature']


forecasting_data_weekly = summary_by_time_weekly(key_data,
                                           'Site_Id',
                                           'Parameter_ParameterDescription')

forecasting_data = forecasting_data_weekly[['Site_Id',
                                  'Date',
                                  'Value_mean']]\
                                  .dropna()

pollutant_df_with_futureframe = forecasting_data\
    .groupby('Site_Id') \
    .future_frame(
        date_column = 'Date',
        length_out  = 20
    )

pollutants_df_dates = pollutant_df_with_futureframe.augment_timeseries_signature(date_column = 'Date')
pollutants_df_dates.head(10)
pollutants_df_dates.glimpse()


df_with_lags = pollutants_df_dates\
    .groupby('Site_Id') \
    .augment_lags(
        date_column  = 'Date',
        value_column = 'Value_mean',
        lags         = [5,6,7,8,9]
    )


df_with_lags.head(5)


lag_columns = [col for col in df_with_lags.columns if 'lag' in col]


df_with_rolling = df_with_lags \
    .groupby('Site_Id') \
    .augment_rolling(
        date_column  = 'Date',
        value_column = lag_columns,
        window  = 4,
        window_func = 'mean',
        threads = 1 # Change to -1 to use all available cores
    ) 


df_with_rolling[df_with_rolling.Site_Id ==329].head(-10)

all_lag_columns = [col for col in df_with_rolling.columns if 'lag' in col]

df_no_nas = df_with_rolling \
    .dropna(subset=all_lag_columns, inplace=False)

df_no_nas.head()

df_no_nas.glimpse()

future = df_no_nas[df_no_nas.Value_mean.isnull()]

train = df_no_nas[df_no_nas.Value_mean.notnull()]

train_columns =  [ 
    'Site_Id'
    , 'Date_year'
    , 'Date_month'
    , 'Date_yweek'
    , 'Date_mweek'
    , 'Date_wday'
    , 'Value_mean_lag_5'
    , 'Value_mean_lag_6'
    , 'Value_mean_lag_7'
    , 'Value_mean_lag_8'
    , 'Value_mean_lag_5_rolling_mean_win_4'
    , 'Value_mean_lag_6_rolling_mean_win_4'
    , 'Value_mean_lag_7_rolling_mean_win_4'
    , 'Value_mean_lag_8_rolling_mean_win_4'
    ]

X = train[train_columns]
y = train[['Value_mean']]

model = RandomForestRegressor(random_state=123)
model = model.fit(X, y)


predicted_values = model.predict(future[train_columns])
future['y_pred'] = predicted_values

train['type'] = 'actuals'
future['type'] = 'prediction'

full_df = pd.concat([train, future])

full_df['Value_mean'] = np.where(full_df.type =='actuals',
                                  full_df.Value_mean,
                                    full_df.y_pred)



predictions_plot = full_df \
    .groupby('Site_Id') \
    .plot_timeseries(
        date_column = 'Date',
        value_column = 'Value_mean',
        color_column = 'type',
        smooth = False,
        smooth_alpha = 0,
        facet_ncol = 2,
        facet_scales = "free",
        y_intercept_color = tk.palette_timetk()['steel_blue'],
        width = 800,
        height = 600,
        engine = 'plotly'
    )
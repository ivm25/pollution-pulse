import numpy as np
import pandas as pd
import sktime
from sktime.forecasting.arima import ARIMA
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
from sktime.forecasting.sarimax import SARIMAX

analysis_data = pd.read_csv('HistoricalObs.csv', 
                   encoding = 'unicode_escape')

analysis_data['Date'] = pd.to_datetime(analysis_data['Date'])


key_data = analysis_data[analysis_data['Parameter_ParameterDescription'] == 'Temperature']




def forecasting_data_daily(temp_data,
                            ):
    """
    Summarizes the data by day for forecasting.
    
    Parameters:
    key_data (DataFrame): The input data containing pollution measurements.
    site_col (str): The column name for site identifiers.
    param_col (str): The column name for parameter descriptions.
    
    Returns:
    DataFrame: A DataFrame with daily summaries of the specified parameters.
    """
  
    forecasting_data_daily = summary_by_time(temp_data,
                                            'Site_Id',
                                                'Parameter_ParameterDescription')

    forecasting_data_daily_processed = forecasting_data_daily[['Date',
                                                               'Site_Id',
                                                               'Value_mean']].dropna()


    # Create heirarchial data for forecasting
    forecasting_data_daily_processed['Date'] = pd.to_datetime(forecasting_data_daily_processed['Date'])

    forecasting_data = forecasting_data_daily_processed.copy()
    forecasting_data = forecasting_data.reset_index()
    forecasting_data['Date'] = forecasting_data['Date'].dt.to_period('D')


    forecasting_data_daily_processed = forecasting_data_daily_processed.reset_index()
    forecasting_data_daily_processed['time_period'] = forecasting_data_daily_processed['Date'].dt.to_period('D')
    temperature_df = forecasting_data_daily_processed\
        .groupby(['Site_Id','time_period']).mean()   
    temperature_df = temperature_df.drop(columns = ['index','Date'])

    # Create a baseline ARIMA model for each site
    r = range(1,31)

    r_list = []
    for i in r:
        r_list.append(i)
    forecaster = ARIMA()
    forecaster.fit(temperature_df,
                    fh=r_list)

    forecaster.predict()

    y_pred = forecaster.predict()


    # Probababilistic forecasting
    y_pred_int = forecaster.predict_interval(
        
        coverage=0.95
    )
    ret = pd.concat([y_pred, y_pred_int],
             axis = 1)
    predictions_with_intervals = ret.reset_index()
    
    predictions_with_intervals.columns = ['Site_Id',
                                           'Date',
                                             'Value_mean',
                                               'Lower Bound',
                                                 'Upper Bound']
    
    # concatanate the dataframes
    predictions_data = pd.concat([forecasting_data,
                                  predictions_with_intervals],
                                  axis = 0)
    

    predictions_data['Date'] = predictions_data['Date'].dt.to_timestamp()

    return predictions_data




def forecasting_data_daily_sarimax(temp_data,
                            ):
    """
    Summarizes the data by day for forecasting.
    
    Parameters:
    key_data (DataFrame): The input data containing pollution measurements.
    site_col (str): The column name for site identifiers.
    param_col (str): The column name for parameter descriptions.
    
    Returns:
    DataFrame: A DataFrame with daily summaries of the specified parameters.
    """
  
    forecasting_data_daily = summary_by_time(temp_data,
                                            'Site_Id',
                                                'Parameter_ParameterDescription')

    forecasting_data_daily_processed = forecasting_data_daily[['Date',
                                                               'Site_Id',
                                                               'Value_mean']].dropna()


    # Create heirarchial data for forecasting
    forecasting_data_daily_processed['Date'] = pd.to_datetime(forecasting_data_daily_processed['Date'])

    forecasting_data = forecasting_data_daily_processed.copy()
    forecasting_data = forecasting_data.reset_index()
    forecasting_data['Date'] = forecasting_data['Date'].dt.to_period('D')


    forecasting_data_daily_processed = forecasting_data_daily_processed.reset_index()
    forecasting_data_daily_processed['time_period'] = forecasting_data_daily_processed['Date'].dt.to_period('D')
    temperature_df = forecasting_data_daily_processed\
        .groupby(['Site_Id','time_period']).mean()   
    temperature_df = temperature_df.drop(columns = ['index','Date'])

    # Create a baseline ARIMA model for each site
    r = range(1,31)

    r_list = []
    for i in r:
        r_list.append(i)
    forecaster_2 = SARIMAX(order = (1,0,0),
                        trend = 'c', 
                        seasonal_order=(1, 0, 0, 4))
    forecaster_2.fit(temperature_df,
                    fh=r_list)

    forecaster_2.predict()

    y_pred = forecaster_2.predict()


    # Probababilistic forecasting
    y_pred_int = forecaster_2.predict_interval(
        
        coverage=0.95
    )
    ret = pd.concat([y_pred, y_pred_int],
             axis = 1)
    predictions_with_intervals = ret.reset_index()
    
    predictions_with_intervals.columns = ['Site_Id',
                                           'Date',
                                             'Value_mean',
                                               'Lower Bound',
                                                 'Upper Bound']
    
    # concatanate the dataframes
    predictions_data = pd.concat([forecasting_data,
                                  predictions_with_intervals],
                                  axis = 0)
    

    predictions_data['Date'] = predictions_data['Date'].dt.to_timestamp()

    return predictions_data
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
from sktime.forecasting.neuralforecast import NeuralForecastLSTM

from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.split import temporal_train_test_split
from databricks.connect.session import DatabricksSession
from pyspark.sql.functions import col

spark = DatabricksSession.builder.serverless().getOrCreate()

t = spark.read.table("workspace.pollution_data.historical_obs_new") 

analysis_data = t.toPandas()



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
                                                               'Value']].dropna()


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
                                             'Value',
                                               'Lower Bound',
                                                 'Upper Bound']
    
    # concatanate the dataframes
    predictions_data = pd.concat([forecasting_data,
                                  predictions_with_intervals],
                                  axis = 0)
    

    predictions_data['Date'] = predictions_data['Date'].dt.to_timestamp()

    return predictions_data




def forecasting_data_daily_sarimax(temp_data,
                                   test_size_days=30  # New parameter for evaluation period
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
                                                               'Value']].dropna()


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
    

    # --- 1. Train-Test Split for Evaluation ---
    # Convert 'time_period' to a proper date/timestamp for splitting
    dates = temperature_df.index.get_level_values('time_period').unique().sort_values()
    split_date = dates[-test_size_days] # Get the start date of the test set

    # Split the data
    y_train = temperature_df.loc[temperature_df.index.get_level_values('time_period') < split_date]
    y_test = temperature_df.loc[temperature_df.index.get_level_values('time_period') >= split_date]

    # The actual horizon for the test set is the length of the dates in y_test
    # This is important for matching the 'fh' parameter in the fit/predict calls
    test_horizon = len(y_test.index.get_level_values('time_period').unique())



    # Create a baseline SARIMAX model for each site
      # --- 2. Fit and Evaluate Model ---
    forecaster = SARIMAX(order = (1,0,0),
                        trend = 'c', 
                        seasonal_order=(1, 0, 0, 4))
    # Fit the model only on the training data
    forecaster.fit(y_train)

    # In-sample prediction (on the test set) for evaluation
    # The 'fh' parameter specifies the forecast horizon relative to the end of the training data
    fh_test = np.arange(1, test_horizon + 1)
    y_pred_test = forecaster.predict(fh=fh_test)

    # --- 3. Calculate Evaluation Metrics ---
    # Extract actual and predicted values, aligning them
    y_test_values = y_test['Value'].values
    y_pred_values = y_pred_test['Value'].values

    # Ensure lengths match before calculating metrics (should match if split is correct)
    if len(y_test_values) != len(y_pred_values):
        print("Warning: Test and prediction lengths do not match. Evaluation skipped.")
        evaluation_metrics = pd.DataFrame() # Return empty DataFrame
    else:
        # Calculate standard metrics
        mae = mean_absolute_error(y_test_values, y_pred_values)
        rmse = np.sqrt(mean_squared_error(y_test_values, y_pred_values))
        
        # Mean Absolute Percentage Error (MAPE) is often useful but requires a specific function
        # Using a custom calculation if not using sktime's metric
        # Avoid division by zero by adding a small epsilon or filtering
        mape = np.mean(np.abs((y_test_values - y_pred_values) / y_test_values)) * 100
        
        # Collect metrics into a DataFrame
        evaluation_metrics = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'MAPE'],
            'Value': [mae, rmse, mape]
        })

     # --- 4. Final Forecast (Out-of-Sample) ---
    # Now, fit the model on the full original dataset to generate the FINAL forecasts
    forecaster.fit(temperature_df)

    # r is the forecast horizon for the future (out-of-sample)
    r = range(1, 31)
    r_list = list(r)
    
    y_pred = forecaster.predict(fh=r_list)

    # Probabilistic forecasting
    y_pred_int = forecaster.predict_interval(coverage=0.95)
    
    # Rest of the prediction processing (same as original)
    ret = pd.concat([y_pred, y_pred_int], axis=1)
    predictions_with_intervals = ret.reset_index()
    
    predictions_with_intervals.columns = ['Site_Id',
                                          'Date',
                                          'Value',
                                          'Lower Bound',
                                          'Upper Bound']
    predictions_with_intervals['Value'] = predictions_with_intervals['Value'].round(2)
    predictions_with_intervals['Lower Bound'] = predictions_with_intervals['Lower Bound'].round(2)
    predictions_with_intervals['Upper Bound'] = predictions_with_intervals['Upper Bound'].round(2)

    # Concatenate the historical data and the future predictions
    # Need to convert historical 'Value_mean' column to be compatible with prediction columns
    historical_data_for_concat = temperature_df.reset_index()
    historical_data_for_concat.columns = ['Site_Id', 'Date', 'Value']
    historical_data_for_concat['Lower Bound'] = np.nan
    historical_data_for_concat['Upper Bound'] = np.nan


    # Ensure the 'Date' column is compatible before concat (Period vs Timestamp)
    predictions_with_intervals['Date'] = predictions_with_intervals['Date'].dt.to_timestamp()
    historical_data_for_concat['Date'] = historical_data_for_concat['Date'].dt.to_timestamp()

    predictions_data = pd.concat([historical_data_for_concat,
                                   predictions_with_intervals],
                                   axis=0, ignore_index=True)

 

    return predictions_data




# def forecasting_data_daily_lstm(temp_data,
#                                    test_size_days=30  # New parameter for evaluation period
#                             ):
#     """
#     Summarizes the data by day for forecasting.
    
#     Parameters:
#     key_data (DataFrame): The input data containing pollution measurements.
#     site_col (str): The column name for site identifiers.
#     param_col (str): The column name for parameter descriptions.
    
#     Returns:
#     DataFrame: A DataFrame with daily summaries of the specified parameters.
#     """
  
#     forecasting_data_daily = summary_by_time(temp_data,
#                                             'Site_Id',
#                                                 'Parameter_ParameterDescription')

#     forecasting_data_daily_processed = forecasting_data_daily[['Date',
#                                                                'Site_Id',
#                                                                'Value']].dropna()


#     # Create heirarchial data for forecasting
#     forecasting_data_daily_processed['Date'] = pd.to_datetime(forecasting_data_daily_processed['Date'])

#     forecasting_data = forecasting_data_daily_processed.copy()
#     forecasting_data = forecasting_data.reset_index()
#     # forecasting_data['Date'] = forecasting_data['Date'].dt.to_period('D')
    

#     forecasting_data_lstm = forecasting_data.copy()
#     forecasting_data_lstm.dropna(inplace=True)
   
    
#     # --- 1. Train-Test Split for Evaluation ---
#     # Convert 'time_period' to a proper date/timestamp for splitting

#     neural_x = forecasting_data_lstm[['Date']]
#     neural_y = forecasting_data_lstm[['Value']]

#     y_train_, y_test_, X_train_, X_test_ = temporal_train_test_split(neural_y,
#                                                                       neural_x, test_size=4)
    

#     model_neural = NeuralForecastLSTM(  
#      max_steps=10, freq=2
#         )
    
    
#     model_neural.fit(y_train_, X=X_train_, fh=[1, 2, 3, 4,6,7,8,9]) 

#     predictions_neural = model_neural.predict(X=X_test_)   
    
#      # concatanate the dataframes
#     predictions_data_neural = pd.concat([forecasting_data,
#                                   predictions_neural],
#                                   axis = 0)
    
  
 

#     return predictions_neural
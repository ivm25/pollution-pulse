import numpy as np
import pandas as pd
import sktime
import pytimetk
from pytimetk import summarize_by_time
import plotly
import os


analysis_data = pd.read_csv('HistoricalObs.csv', 
                   encoding = 'unicode_escape')

analysis_data['Date'] = pd.to_datetime(analysis_data['Date'])


def summary_by_time(df, key):

    data_by_category = df\
                    .groupby([key])\
                    .summarize_by_time(
                        date_column = 'Date',
                        value_column = 'Value',
                        freq = "D",
                        agg_func = 'mean',
                        wide_format = False,
                        engine = "polars"
                    )
    
    return data_by_category



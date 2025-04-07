import numpy as np
import pandas as pd
import sktime
import pytimetk
from pytimetk import summarize_by_time
from pytimetk import anomalize
import plotly
import os





def summary_by_time(df, key1,key2):

    data_by_category = df\
                    .groupby([key1,key2])\
                    .summarize_by_time(
                        date_column = 'Date',
                        value_column = 'Value',
                        freq = "D",
                        agg_func = 'mean',
                        wide_format = False,
                        engine = "polars"
                    )
    
    return data_by_category


def summary_by_time_weekly(df, key1,key2):

    data_by_category = df\
                    .groupby([key1,key2])\
                    .summarize_by_time(
                        date_column = 'Date',
                        value_column = 'Value',
                        freq = "W",
                        agg_func = 'mean',
                        wide_format = False,
                        engine = "polars"
                    )
    
    return data_by_category


def anamoly_detection(df, key1, key2):

    anomalised_df = df.groupby([key1, key2], sort = False)\
                      .anomalize(date_column = "Date",
                                 value_column = 'Value_mean',
                                 period = 6)
    
    return anomalised_df

import numpy as np
import pandas as pd
import sktime
import pytimetk
from pytimetk import summarize_by_time
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



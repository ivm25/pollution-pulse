import numpy as np
import pandas as pd
import sktime
import pytimetk
from pytimetk import summarize_by_time
from pytimetk import anomalize
import plotly
import os

from data_manipulation.data_wrangling import summary_by_time,air_category_summary_by_time, summary_by_time_weekly, anamoly_detection,data_for_classification
from datetime import datetime, timedelta
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.dists_kernels import FlatDist, ScipyDist
from sklearn.model_selection import train_test_split

analysis_data = pd.read_csv('HistoricalObs.csv', 
                   encoding = 'unicode_escape')

today = datetime.now()
year_ago = today - timedelta(days = 365)

last_year = year_ago.year
this_year = today.year

this_year_start = datetime(this_year,1,1)
last_year_start = datetime(last_year, 1, 1)

analysis_data['Date'] = pd.to_datetime(analysis_data['Date'])

analysis_data_this_year = analysis_data[analysis_data['Date'] > this_year_start]


manipulated_data = data_for_classification(analysis_data_this_year,
                                            var_to_model = 'PM10')

summary_data = air_category_summary_by_time(manipulated_data,
                                           'Site_Id',
                                           'Hour',
                                           'Parameter_ParameterDescription',
                                           'AirQualityCategory')

summary_data = summary_data.dropna()

def categorical_y_values(target):

    if target == 'GOOD':
        return 1
    elif target == 'FAIR':
        return 2
    elif target == 'POOR':
        return 0
    elif target == 'VERY POOR':
        return 4
    else:
        return 5


summary_data["AirQualityCategory"] = summary_data["AirQualityCategory"].apply(categorical_y_values)  



x = summary_data.drop(columns=['AirQualityCategory',
                                   'Parameter_ParameterDescription',
                                   ], axis = 1)

x_multi = x.set_index(['Site_Id','Date','Hour'])

x_array = np.array(x_multi)

y = summary_data['AirQualityCategory']

y_array = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(x_array,
                                                    y_array,
                                                    test_size=0.33,
                                                     random_state=42)







# example 1 - 3-NN with simple dynamic time warping distance (requires numba)
clf = KNeighborsTimeSeriesClassifier(n_neighbors=3)

# example 2 - custom distance:
# 3-nearest neighbour classifier with Euclidean distance (on flattened time series)
# (requires scipy)
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.dists_kernels import FlatDist, ScipyDist

eucl_dist = FlatDist(ScipyDist())
clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, 
                                     distance=eucl_dist)

clf.get_params()


# step 2 - fit the classifier
clf.fit(X_train, y_train)

# step 3 - predict labels on new data

y_pred = clf.predict(X_test)

# for simplest evaluation, compare ground truth to predictions
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

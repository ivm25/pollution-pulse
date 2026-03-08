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
from sktime.split import temporal_train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data_manipulation.data_wrangling import data_for_classification, air_category_summary_by_time
from sklearn.preprocessing import LabelEncoder  


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


    
encoder = LabelEncoder()



def prepare_data(analysis_data, var_to_model='PM10'):
    """
    Performs data filtering, feature engineering, and splits data into
    features (X) and target (y) arrays.
    """
    # 1. Date Filtering
    today = datetime.now()
    this_year = today.year
    this_year_start = datetime(this_year, 1, 1)

    analysis_data['Date'] = pd.to_datetime(analysis_data['Date'])
    analysis_data_this_year = analysis_data[analysis_data['Date'] > this_year_start].copy()

    # 2. Pre-processing and Summarization (Mocking external functions)
  
    # are correctly imported/defined.
    # For this example, we'll assume they return a DataFrame similar to your original 'summary_data'.
    
    # Mocking data_for_classification and air_category_summary_by_time output structure
    
    # Since the full logic of external functions isn't here, we'll assume a structure:
    # simulated_manipulated_data = data_for_classification(analysis_data_this_year, var_to_model=var_to_model)
    # summary_data = air_category_summary_by_time(
    #     simulated_manipulated_data, 'Site_Id', 'Hour', 'Parameter_ParameterDescription', 'AirQualityCategory'
    # )
    
    # --- Using your existing code logic for the data frame structure ---
    
    # Mocking the result of data_for_classification and air_category_summary_by_time
    # This block assumes the external functions are defined in your environment
  
   
    manipulated_data = data_for_classification(analysis_data_this_year, var_to_model=var_to_model)
    summary_data = air_category_summary_by_time(manipulated_data, 'Site_Id', 'Hour', 'Parameter_ParameterDescription', 'AirQualityCategory'
    )
    # except ImportError:
    #     print("Warning: External functions 'data_for_classification' and 'air_category_summary_by_time' not available. Skipping this step.")
    #     # Fallback for demonstration: Assume summary_data is already loaded or partially processed
    #     summary_data = analysis_data_this_year.copy() 
    #     # Add mock columns if they don't exist for the rest of the logic to work
    #     if 'Value_mean' not in summary_data.columns:
    #          summary_data['Value_mean'] = summary_data.get('Value', 0) # Use 'Value' or a default
    #     if 'AirQualityCategory' not in summary_data.columns:
    #         # Simple mock for 'AirQualityCategory' based on 'Value' or 'Site_Id'
    #          summary_data['AirQualityCategory'] = np.random.choice(['GOOD', 'FAIR', 'POOR'], size=len(summary_data))
        
    summary_data = summary_data.dropna()

    # 3. Categorical Encoding
    # Ensure only normal spaces are used here (remove any non-breaking spaces)
    summary_data["AirQualityCategory_Encoded"] = encoder.fit_transform(summary_data["AirQualityCategory"])

    # 4. Feature and Target Selection
    keycols_for_classification = ['Date', 'Value_mean', 'AirQualityCategory_Encoded']
    classification_data = summary_data[keycols_for_classification]

    pm10_classification_features = classification_data[['Date', 'Value_mean']]
    pm10_classification_target = classification_data['AirQualityCategory_Encoded']

    # 5. Final Array Conversion for sktime
    # sktime works better with time series data indexed by time, but for array splitting:
    pm10_classification_features.set_index(['Date'], inplace=True)

    features_array = np.array(pm10_classification_features)
    target_array = np.array(pm10_classification_target)
    X_train,X_test, y_train, y_test = temporal_train_test_split(features_array, target_array, test_size = 0.30,
                                                    )
    # return features_array, target_array
   
    return X_train, X_test, y_train, y_test


def train_time_series_classifier(X_train, y_train, n_neighbors=3):
    """
    Initializes and trains a KNeighborsTimeSeriesClassifier.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target labels.
        n_neighbors (int): The number of nearest neighbors to use.
        
    Returns:
        KNeighborsTimeSeriesClassifier: The fitted classifier.
    """
    print(f"Starting model training with n_neighbors={n_neighbors}...")
    
    # Initialize the classifier (default distance is usually DTW which is good for TS)
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors)
    
    # Fit the classifier
    clf.fit(X_train, y_train)
    
    print("Model training complete. ✅")
    return clf


def evaluate_classifier(clf, X_test, y_test):
    """
    Predicts labels and calculates evaluation metrics (accuracy, report, 
    confusion matrix).
    
    Args:
        clf (KNeighborsTimeSeriesClassifier): The trained classifier.
        X_test (np.array): Testing features.
        y_test (np.array): Testing target labels (ground truth).
    """
    print("\n--- Model Evaluation ---")
    
    # 1. Predict labels on new data
    y_pred = clf.predict(X_test)

    # Define target names corresponding to the numerical mapping in categorical_y_values
    # Note: Ensure these align with your mapping: 0=POOR, 1=GOOD, 2=FAIR, 4=VERY POOR, 5=EXTREMELY POOR
    target_names = ['POOR', 'GOOD', 'FAIR', 'VERY POOR', 'EXTREMELY POOR'] 

    # 2. Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    # 3. Detailed classification report
    print("Classification Report:\n")
    # Get unique labels from y_test and sort them to ensure correct alignment with target_names
    # This is crucial because some categories might not be present in the test set.
    test_labels = sorted(np.unique(y_test))
    
    # Filter target names to only include those corresponding to the labels present in y_test
    # This prevents errors if a label (e.g., 3 or 5) is not present in the test set.
    # The actual mapping for labels: 0=POOR, 1=GOOD, 2=FAIR, 4=VERY POOR, 5=EXTREMELY POOR
    # Assuming labels 0, 1, 2, 4, 5 are the only possible ones, and target_names is indexed 0 to 4.
    # Since the mapping is not sequential (missing 3), we'll rely on labels=sorted(np.unique(y_test)) 
    # to handle the indices correctly, but target_names MUST correspond to the integer values.
    
    # Create a simple list mapping integers back to names based on your function
    label_map = {0: 'POOR', 1: 'GOOD', 2: 'FAIR', 3: 'VERY POOR', 4: 'EXTREMELY POOR'}
    # Filter target names based on actual labels in y_test
    names_for_report = [label_map[label] for label in test_labels]
    
    report = classification_report(
        y_test, y_pred, 
        labels=test_labels, 
        target_names=names_for_report, 
        zero_division=0
    )
    print(report)

    # 4. Confusion Matrix Visualization
    print("Generating Confusion Matrix... 📊")
    cm = confusion_matrix(y_test, y_pred, labels=test_labels)
    
    # Use the same names_for_report for display_labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names_for_report)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix for Air Quality Classification')
    plt.show()


    
# summary_data["AirQualityCategory"] = summary_data["AirQualityCategory"].apply(categorical_y_values)  

# keycols_for_classification = ['Date', 
#                               'Site_Id',
                               
#                                     'Value_mean',
#                                     'AirQualityCategory']

# classification_data  = summary_data[keycols_for_classification]

# x = summary_data.drop(columns=['AirQualityCategory',
#                                    'Parameter_ParameterDescription',
#                                    ], axis = 1)




# x_multi = x.set_index(['Site_Id','Date','Hour'])

# x_array = np.array(x_multi)

# y = summary_data['AirQualityCategory']

# y_array = np.array(y)
# pm10_classification_features = classification_data[['Date', 'Value_mean']]
# pm10_classification_target = classification_data['AirQualityCategory']

# pm10_classification_features.set_index(['Date'],
#                                         inplace=True)

# features_array = np.array(pm10_classification_features)
# target_array = np.array(pm10_classification_target)


# from sktime.split import temporal_train_test_split

# X_train,X_test, y_train, y_test = temporal_train_test_split(features_array,
#                                                              target_array,
#                                                                test_size = 0.30,
#                                                     )


# # example 1 - 3-NN with simple dynamic time warping distance (requires numba)
# clf = KNeighborsTimeSeriesClassifier(n_neighbors=3)


# clf.fit(X_train, y_train)









# eucl_dist = FlatDist(ScipyDist())
# clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, 
#                                      distance=eucl_dist)

clf.get_params()


# step 2 - fit the classifier
clf.fit(X_train, y_train)

# step 3 - predict labels on new data

y_pred = clf.predict(X_test)

# for simplest evaluation, compare ground truth to predictions

# Evaluation Metrics

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

# 2. Print a detailed classification report
print("Classification Report:\n")
# Create a list of target names based on your categorical_y_values mapping
target_names = ['POOR', 'GOOD', 'FAIR', 'VERY POOR', 'EXTREMELY POOR'] 
# Use the unique values in y_test as labels for the report
report = classification_report(y_test, y_pred, labels=sorted(np.unique(y_test)), 
                               target_names=target_names, zero_division=0)
print(report)


cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y_test)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Air Quality Classification')
plt.show()

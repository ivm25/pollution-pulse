import os
import sys
import requests
import logging
import urllib
import datetime as dt
import pandas as pd


import json
from pandas import json_normalize

class aqms_api_class(object):


    def __init__(self, ):

        self.logger = logging.getLogger('aqms_api_class')
        self.url_api = "https://data.airquality.nsw.gov.au"
        self.headers = {'content-type': 'application/json', 'accept': 'application/json'}
        self.get_observations = 'api/Data/get_Observations'
        return
    
    def get_Obs(self, ObsRequest):


        query = urllib.parse.urljoin(self.url_api, self.get_observations)
        response = requests.post(url = query,
                                 data = json.dumps(ObsRequest),
                                 headers = self.headers)
        
        return response.json()
    
    def ObsRequest_init(self, ):

        ObsRequest = {}

        ObsRequest['Parameters'] = ['PM10']
        ObsRequest['Sites'] = [336,4330,2330, 7330,3330,329,5330]
        StartDate = dt.date(2021,11,25)
        EndDate = dt.date(2025,1,21)
        ObsRequest['StartDate'] = StartDate.strftime('%Y-%m-%d')
        ObsRequest['EndDate'] = EndDate.strftime('%Y-%m-%d')
        ObsRequest['Categories'] = ['Averages']
        ObsRequest['SubCategories'] = ['Hourly']
        ObsRequest['Frequency'] = ['Hourly average']
        
  
        return ObsRequest
    

if __name__ == '__main__':
    AQMS = aqms_api_class()
    ObsRequest = AQMS.ObsRequest_init()
    AllHistoricalObs = AQMS.get_Obs(ObsRequest)
    
    if AllHistoricalObs:
        print(json.dumps(AllHistoricalObs, indent=4))

        df = json_normalize(AllHistoricalObs, sep='_')
        print(df.head())
        # # Save DataFrame to CSV (optional)
        df.to_csv('HistoricalObs.csv',
                   index=False,
                    encoding='ISO-8859-1')


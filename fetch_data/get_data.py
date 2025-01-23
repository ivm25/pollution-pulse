import os
import sys
import pandas as pd

import requests
import logging
import urllib
import datetime as dt
import json


class aqms_api_class(object):


    def __init__(self,):
        self.logger = logging.getLogger(__file__)
        self.url_api = "https://data.airquality.nsw.gov.au"
        self.headers = {'content-type':'application/json', 'accept': 'application/json'}
        self.get_site_url = "/api/Data/get_SiteDetails"
        return
    
    def get_site_details(self,):

        query = urllib.parse.urljoin(self.url_api,self.get_site_url)
        response = response.get(url = query, data = '')
        return response
    
if __name__=='__main__':
    AQMS = aqms_api_class()

    AllSites = AQMS.get_site_details()

    f = open('SiteDetails.txt', 'w')
    for item in AllSites:
        item = item.decode("ISO-8859-1")
        f.write(str(item) + '\n')
    f.close()





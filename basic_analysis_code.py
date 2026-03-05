# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 23:10:41 2025

@author: Rico's Account
"""

import pandas as pd

def filter_data(data, area_thresh):
    filtered_data = data[data['Area']>area_thresh]
    
    return filtered_data

def analyze_data(path, area_thresh):
    data = pd.read_csv(path)
    filtered_data = filter_data(data, area_thresh)
    avg_major = filtered_data['Major'].mean()
    avg_minor = filtered_data['Minor'].mean()
    
    return avg_major, avg_minor

#soft_path = 'C:/Users/Rico's Account/Downloads'
stiff_path = "C:/Users/Rico's Account/Downloads/1_results.csv"

area_thresh = 0.3

#avg_soft_FA_length = analyze_data(soft_path, area_thresh)
avg_stiff_FA_length, avg_stiff_FA_width = analyze_data(stiff_path, area_thresh)

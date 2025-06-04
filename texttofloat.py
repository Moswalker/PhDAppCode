"""
This code was written by Lucas Moshoej as a part of the Master thesis: Investment Optimization of an Energy Capacity Portfolio using Stochastic Modelling
Fixing text strings
"""
import pandas as pd

def texttofloat(data):
    #The PV is text strings, so gotta fix that, 
    #commas are replaced  with periods for decimal separators and then converted to numeric
    data['Pv'] = pd.to_numeric(data['Pv'].str.replace(',', '.'), errors='coerce')  # Handle commas as decimals

    #check for NaN values
    #print(data['Pv'].isna().sum())  # Number of NaN values

    # Handle NaNs if any
    data['Pv'].fillna(0, inplace=True)
    
    return data
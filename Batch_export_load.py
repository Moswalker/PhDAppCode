"""
Loading data from batch export
"""
import pandas as pd


def load_data(filename):

    #load data 
    data = pd.read_csv(filename, delimiter=';')

    # Drop unnecessary columns
    columns_to_ignore = ['UserName', 'ModelName', 'Studyname']
    data = data.drop(columns=columns_to_ignore, errors='ignore')

    return data #scenario_groups


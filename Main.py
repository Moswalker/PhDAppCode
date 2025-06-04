"""
This code was written by Lucas Moshoej as a part of the Master thesis: Investment Optimization of an Energy Capacity Portfolio using Stochastic Modelling

This file is used for calling cashflow and capacity data visualization functions
"""
#%%
"Libraries"
import os
import numpy as np
import pandas as pd

#Correct the path, necessary for the program to find the correct modules
#Finding folder where script is located
current_folder = os.path.dirname(os.path.abspath(__file__))

#Setting folder as the working directory
os.chdir(current_folder)


#%% Import all functions defined for data loading and visualization 

from Batch_export_load import load_data # Sets delimiters and groups to ignore
from texttofloat import texttofloat #
from Costplots import costplot
from Capplot import Cap_bar_plot, CAPplot 
from Cashflow_functions import Cash_flow_bar_stoch_avg_with_stack, Cash_flow_bar_stoch_avg2, Cash_flow_bar_det

#Setting the main scenario you would like to analyze, relevant through some of the functions
#3 options: "timeseries_stoch_stochastic", "timeseries_stoch_spines", "deterministic"
scenarioChosen = ["timeseries_stoch_stochastic"]

#%% 
"Cost data (only) plots"
filename = "Cost_related_attributes.csv"
Cost_data1 = load_data(filename)
Cost_data = texttofloat(Cost_data1) 

#Plotting cost values
variances_by_year, costs_data, income_data, milestone_years = costplot(Cost_data, scenarioChosen)

#%%
"Capacity plots"

#Runs range from 20 to 30, and describe what sensitivity analysis is being performed
run = '27' 

#%Cap table, Cash flow and plots
file_path = "Capacity values.csv"

#All 3 model types
scenarios = ["timeseries_stoch_stochastic","timeseries_stoch_spines", 'deterministic']

#Looping over all 3 scenarios, to get the installed capacities in each
for Scenarioname in scenarios:
    CAPplot(file_path, Scenarioname, run)


#%% 
"Cashflow plots - This cell is relevant for PhD"
#Runs range from 20 to 30, and describe what sensitivity analysis is being performed - list is stored in separate folder.
run = '27'
file_path = 'Cost_related_attributes.csv'

#Stochresults provide the categorized costs along with the min and max cashflow for the future scenarios
Stochresults, results = Cash_flow_bar_stoch_avg_with_stack(file_path, run)

#%%Stochresults2 provide the average costs with error bars across the future scenarios, with min and max cashflows
Stochresults2, results = Cash_flow_bar_stoch_avg2(file_path, run)

#%%Deterministic only has one future scenario, but otherwise provides the broken down costs same as Stochresults
detresults = Cash_flow_bar_det(file_path, run)


#%%
#Convert relevant values from the Stochresults to an excel file.

Scenarios = ['timeseries_stoch_stochastic','timeseries_stoch_spines', 'deterministic']

Values = ["Std_value","NPV", "IRR","Crossing year"]
Xcelname = ["Stoch","Spine", 'Det']
i = 0
writer = pd.ExcelWriter(f'Results_{run}.xlsx', engine='xlsxwriter')

for Valuename in Values:
    #Stochresults[i]
    
    for Scenname, Excelname in zip(Scenarios, Xcelname):
       if Excelname == 'Det':
           #if Valuename == "Std_value":
               continue
           #elif Valuename == "NPV":
           
          #     Dataframe = (pd.DataFrame({Valuename:detresults[0]}))
          #     Dataframe.to_excel(writer, sheet_name=f'{Valuename}_{Excelname}')
          # elif Valuename == "IRR":
          #     Dataframe = (pd.DataFrame({Valuename:detresults[1]}))
          #     Dataframe.to_excel(writer, sheet_name=f'{Valuename}_{Excelname}')
          # else:
           #    Dataframe = (pd.DataFrame({Valuename:detresults[2]}))
           #    Dataframe.to_excel(writer, sheet_name=f'{Valuename}_{Excelname}')
        #print(Scenname)
       #print(Excelname)
       else:
           Dataframe = (pd.DataFrame({Valuename:Stochresults[i][Scenname]}))
           Dataframe.to_excel(writer, sheet_name=f'{Valuename}_{Excelname}')
       
    i=+1
Dataframe = pd.DataFrame([{'NPV':detresults[0],'IRR':detresults[1], 'Cross':detresults[2]}])
Dataframe.to_excel(writer, sheet_name='Det_all')
       #Open the file for writing (this ensures it's handled correctly)
writer.close()#with open(f'Std_value_{name}.txt', 'w') as file:
       #    for item in WThis:
       #    file.write(str(item) + "\n")









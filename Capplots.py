"""
This code was written by Lucas Moshoej as a part of the Master thesis: Investment Optimization of an Energy Capacity Portfolio using Stochastic Modelling

Installed capacity plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

S = 10
M= 12
L = 14


def CAPplot(file_path, scenarioChosen, run):

    # Load the data
    Capdata = pd.read_csv(file_path, delimiter=';', decimal=",")
    years = sorted(Capdata['Period'].unique())
    sows = sorted(Capdata['Sow'].unique())
    if scenarioChosen == 'timeseries_stoch_stochastic':
        name = 'Stochastic'
    elif scenarioChosen == 'timeseries_stoch_spines':
        name = 'Spines'
    else:
        name = 'Deterministic'
    
    # Define process-specific colors and legend names
    colors = {
        "IMPELC-DKW": "tab:brown",
        "EXPELC-DKW": "tab:purple",
        "MINGAS1": "tab:red",
        "ELERNWINDON": "limegreen",
        "ELERNWINDOFF8": "lightseagreen",
        "ELERNWINDOFF20": "deepskyblue",
        "ELERNWINDOFF30": "teal",
        "ELCTEGAS01": "black",
        "ELERNWSUN01": "tab:orange",
        "ELERNWSUN02": "gold"
    }
    
    legend_names = {
        "IMPELC-DKW": "Import",
        "EXPELC-DKW": "Export",
        "MINGAS1": "Gas Import",
        "ELERNWINDON": "Onshore WTG",
        "ELERNWINDOFF8": "Offshore 8.4 WTG",
        "ELERNWINDOFF20": "Offshore 20 WTG",
        "ELERNWINDOFF30": "Offshore 30 WTG",
        "ELCTEGAS01": "Gas turbine",
        "ELERNWSUN01": "PV",
        "ELERNWSUN02": "PV Bi"
    }

    for year in years:
        # Filter for the selected scenario and year
        Capfilter = Capdata[
            (Capdata['Attribute'] == 'VAR_Cap') &
            (Capdata['Scenario'] == scenarioChosen) &
            (Capdata['Period'] == year)
        ]
        #print(Capfilter)

        if set(Capfilter['Sow']) == {'-'}:  # Deterministic plot
            processes = Capfilter['Process']
            capacities = Capfilter['Pv']
            colors_list = [colors.get(process, "gray") for process in processes]  # Default to gray if process not in colors
            labels = [legend_names.get(process, process) for process in processes]  # Use new legend names

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(labels, capacities, color=colors_list, edgecolor='black')
            #ax.set_xlabel('Process', fontsize=M)
            ax.set_ylabel('Capacity (GW)', fontsize=M)
            #ax.set_title(f'Capacity {name} {year}')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=30, ha='right',fontsize=M)
            plt.yticks(fontsize=M)
            plt.tight_layout()
            #Uncomment to save graph
            #if year == 2045:
                #plt.savefig(f'{run} Capacity {name}.png', bbox_inches='tight')
            plt.show()
        #else:
        elif scenarioChosen=='timeseries_stoch_stochastic': # Stochastic plot
            # Pivot the data to get processes as columns and Sows as rows
            grouped = Capfilter.pivot(index='Sow', columns='Process', values='Pv').fillna(0)
            sows = grouped.index
            processes = grouped.columns

            x = np.arange(len(sows))  # Positions for Sows on the x-axis
            width = 0.2  # Width of bars

            fig, ax = plt.subplots(figsize=(12, 6))
            for i, process in enumerate(processes):
                color = colors.get(process, "gray")  # Default to gray if process not in colors
                label = legend_names.get(process, process)  # Use new legend names
                ax.bar(
                    x + i * width, 
                    grouped[process].values,  # Ensure this is an array
                    width, 
                    label=label, 
                    color=[color] * len(sows)  # Ensure color is a list
                )

            ax.set_xlabel('State of the World (Sow)',fontsize=M)
            ax.set_ylabel('Capacity (GW)',fontsize=M)
            #ax.set_title(f'Capacity {name} {year}')
            ax.set_xticks(x + width * (len(processes) - 1) / 2)  # Center ticks under grouped bars
            ax.set_xticklabels(sows,fontsize=M)
            plt.yticks(fontsize=M)
            #plt.yticks(fontsize=S)
            ax.legend(loc='upper left',fontsize=L)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            if year == 2045:
                plt.savefig(f'{run} Capacity {name}.png', bbox_inches='tight')
            plt.show()
        else: 
            #print(Capfilter)
            #Spines = Capfilter['Sow']='1'
            processes = Capfilter['Process']
            print(processes)
            capacities = Capfilter['Pv']
            colors_list = [colors.get(process, "gray") for process in processes]  # Default to gray if process not in colors
            labels = [legend_names.get(process, process) for process in processes]  # Use new legend names
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(labels, capacities, color=colors_list, edgecolor='black')
            #ax.set_xlabel('Process', fontsize=M)
            ax.set_ylabel('Capacity (GW)', fontsize=M)
            ax.set_title(f'Capacity {name} {year}')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=30, ha='right',fontsize=M)
            plt.yticks(fontsize=M)
            plt.tight_layout()
            #Uncomment to save graph
            #if year == 2045:
            #    plt.savefig(f'{run} Capacity {name}.png', bbox_inches='tight')
            plt.show()
    return

#%%
def Cap_bar_plot(file_path, run):
    # Load the data
    data = pd.read_csv(file_path, delimiter=';', decimal=",")
    # Ensure necessary columns are numeric
    data['Sow'] = pd.to_numeric(data['Sow'], errors='coerce')
    data['Pv'] = pd.to_numeric(data['Pv'].astype(str).str.replace(',', '.'), errors='coerce')
    data['Period'] = pd.to_numeric(data['Period'], errors='coerce')

    # Filter for relevant attribute
    data = data[data['Attribute'] == 'VAR_Cap']

    # Define colors for technologies (processes)
    colors = {
        "IMPELC-DKW": "tab:brown",
        "EXPELC-DKW": "tab:purple",
        "MINGAS1": "tab:red",
        "ELERNWINDON": "limegreen",
        "ELERNWINDOFF8": "lightseagreen",
        "ELERNWINDOFF20": "deepskyblue",
        "ELERNWINDOFF30": "teal",
        "ELCTEGAS01": "tab:olive",
        "ELERNWSUN01": "tab:orange",
    }

    # Iterate through scenarios
    scenarios = data['Scenario'].unique()
    for scenario in scenarios:
        scenario_data = data[data['Scenario'] == scenario]

        # Handle Spines: only use SOW = 1
        if scenario == 'timeseries_stoch_spines':
            scenario_data = scenario_data[scenario_data['Sow'] == 1]

        # Group data by Period and Process, calculate mean/std if stochastic
        if scenario == 'timeseries_stoch_stochastic':
            grouped = scenario_data.groupby(['Period', 'Process'])
            mean_std_data = grouped['Pv'].agg(['mean', 'std']).reset_index()
        else:  # For deterministic and spines, use direct values
            mean_std_data = scenario_data.copy()
            mean_std_data['mean'] = mean_std_data['Pv']
            mean_std_data['std'] = 0

        # Scenario naming
        if scenario == 'timeseries_stoch_stochastic':
            name = 'Stochastic'
        elif scenario == 'timeseries_stoch_spines':
            name = 'Spines'
        elif scenario == 'deterministic':
            name = 'Deterministic'

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get unique, sorted periods and map them to x-axis positions
        periods = sorted(mean_std_data['Period'].unique())
        period_positions = {period: i for i, period in enumerate(periods)}
        width = 0.8 / len(mean_std_data['Process'].unique())  # Bar width

        for i, process in enumerate(mean_std_data['Process'].unique()):
            process_data = mean_std_data[mean_std_data['Process'] == process]
            x = [period_positions[period] for period in process_data['Period']]
            ax.bar(
                np.array(x) + i * width,
                process_data['mean'],
                width,
                label=process,
                yerr=process_data['std'],
                color=colors.get(process, "tab:gray"),  # Default to gray if process not in colors
                capsize=5
            )

        # Formatting the plot
        ax.set_title(f"{name}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Investment (GW)")
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, rotation=45)  # Fix issue with x-ticks
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        #plt.savefig(f'InstalledCap_{name}_{run}.png', bbox_inches='tight')
        plt.tight_layout()
        # Save or show the plot
        #plt.savefig(f'InstalledCap_{name}_{run}.png', bbox_inches='tight')
        #plt.savefig(f'InstalledCap_{name}_{run}', bbox_inches='tight')
        plt.show()

    return


#%%Work in progress
def latexCAP(Capdata, scenarioChosen):
    Capfilter = Capdata[(Capdata['Attribute']=='VAR_Cap') & (Capdata['Scenario']==scenarioChosen[0])]

    #Found guide online: https://www.youtube.com/watch?v=gLalZyodYqs
    latex_table = Capfilter[['Process', 'Pv', 'Period', 'Sow']].to_latex(
        index=False,  # Do not include the index of the dataFrame. did that at first
        header=['Technology', 'Capacity (GW)', 'Year', 'SOW'],  
        caption="VAR_Cap Values for 2040",  # add caption. 
        label="tab:var_cap_2040"  # for referencing in LaTeX
        )

    #get the table to copy paste.
    print(latex_table)
    return latex_table
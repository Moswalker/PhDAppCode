"""
Made by Lucas Moshøj as a part of the Master thesis: Investment Optimization of an Energy Capacity Portfolio using Stochastic Modelling
All up to date and relevant cashflow functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

# Early defined text sizes
S = 10
M = 12
L = 14

#%% First function breaks down the costs into stackable bars and provides a cashflow overview
def Cash_flow_bar_stoch_avg_with_stack(filepath,run):

    data = pd.read_csv(filepath, delimiter=';', decimal=",")
    data['Period'] = pd.to_numeric(data['Period'], errors='coerce')  
    data['Vintage'] = pd.to_numeric(data['Vintage'], errors='coerce')
    data['Pv'] = pd.to_numeric(data['Pv'], errors='coerce')
    data['Sow'] = pd.to_numeric(data['Sow'], errors='coerce')  

    scenarios = ['timeseries_stoch_stochastic', 'timeseries_stoch_spines']
    std_dev_results = {}
    npv_results = {}
    irr_results = {}
    zero_crossing_years = {}
    total_results = {}
    for scenario in scenarios:
        print(scenario)
        scenario_data = data[data['Scenario'] == scenario]
        
        unique_sows = scenario_data['Sow'].dropna().unique()
        results = {}

        period_to_years = {
            2025: range(2022, 2028),
            2030: range(2028, 2033),
            2035: range(2033, 2038),
            2040: range(2038, 2043),
            2045: range(2043, 2048)
        }

        for sow in unique_sows:
            sow_data = scenario_data[scenario_data['Sow'] == sow]
            
            costs = {}
            income = {}
            cumulative_cash_flow = {}
            # Create a dictionary for cost breakdown by attribute
            cost_components = {attr: {} for attr in ['Cost_Act', 'Cost_Comx', 'Cost_Fom', 'Cost_Flo','Cost_Inv' ]}
            
            for _, row in sow_data.iterrows():
                attribute = row['Attribute']
                process = row['Process']
                period = row['Period']
                vintage = row['Vintage']
                value = row['Pv']

                if attribute in ['Cost_Act', 'Cost_Comx', 'Cost_Fom', 'Cost_Flo']:
                    if attribute == 'Cost_Flo' and process == 'EXPELC-DKW':
                        # For income, do not add to cost_components
                        yearly_value = (-value) #/ (3 if period in (2030, 2022, 2025) else len(period_to_years[period]))
                        for year in (range(2030, 2033) if period in (2030,2022,2025) else period_to_years[period]):
                            income[year] = income.get(year, 0) + yearly_value
                    else:
                        yearly_value = (-value) #/ (3 if period == 2030 else len(period_to_years[period]))
                        for year in (range(2030, 2033) if period == 2030 else period_to_years[period]):
                            costs[year] = costs.get(year, 0) + yearly_value
                            # Also add to cost_components breakdown
                            cost_components[attribute][year] = cost_components[attribute].get(year, 0) + yearly_value
                elif attribute == 'Cost_Inv':
                    yearly_value = (-value) #/ (3 if period == 2030 else len(period_to_years[period]))
                    #print(yearly_value)
                    for year in (range(2030, 2033) if period == 2030 else period_to_years[period]):
                        costs[year] = costs.get(year, 0) + yearly_value
                        cost_components[attribute][year] = cost_components[attribute].get(year, 0) + yearly_value
            all_years = sorted(set(costs.keys()).union(set(income.keys())))
            yearly_costs, yearly_income, cumulative_cash_flow_values, cash_flow_normal = [], [], [], []
            total_cash_flow = 0

            for year in all_years:
                cost = costs.get(year, 0)
                income_value = income.get(year, 0)
                cash_flow = income_value + cost
                total_cash_flow += cash_flow

                yearly_costs.append(cost)
                yearly_income.append(income_value)
                cumulative_cash_flow_values.append(total_cash_flow)
                cash_flow_normal.append(cash_flow)
            results[sow] = {
                "years": all_years,
                "costs": yearly_costs,
                "income": yearly_income,
                "cash_flow": cumulative_cash_flow_values,
                "cash_flow2": cash_flow_normal,
                "cost_components": cost_components
            }
        #
        npv_values = {sow: sum(results[sow]["cash_flow2"]) for sow in unique_sows}
        irr_values = {sow: npf.irr(results[sow]["cash_flow2"]) for sow in unique_sows}
        #
        npv_results[scenario] = npv_values
        irr_results[scenario] = irr_values
        
        all_years = sorted(set().union(*(res["years"] for res in results.values())))
        avg_costs, min_cash_flow, max_cash_flow, std_costs = [], [], [], []
        
        # Compute stacked average costs for each cost attribute per year across SOWs
        stacked_avg_costs = {attr: [] for attr in ['Cost_Act', 'Cost_Comx', 'Cost_Fom', 'Cost_Flo','Cost_Inv']}
        for year in all_years:
            for attr in ['Cost_Act', 'Cost_Comx', 'Cost_Fom', 'Cost_Flo','Cost_Inv']:
                attr_values = []
                for sow in unique_sows:
                    if year in results[sow]['cost_components'][attr]:
                        attr_values.append(results[sow]['cost_components'][attr][year])
                    else:
                        attr_values.append(0)
                stacked_avg_costs[attr].append(np.mean(attr_values))
        
        # Sum the stacked averages to get overall average cost
        for i in range(len(all_years)):
            avg_costs.append(sum(stacked_avg_costs[attr][i] for attr in ['Cost_Act', 'Cost_Comx', 'Cost_Fom', 'Cost_Flo']))
            # Compute std deviation of overall costs across SOWs
            year_costs = []
            for sow in unique_sows:
                if all_years[i] in results[sow]['years']:
                    idx = results[sow]['years'].index(all_years[i])
                    year_costs.append(results[sow]['costs'][idx])
                else:
                    year_costs.append(0)
            std_costs.append(np.std(year_costs))
        
        zero_crossing_years[scenario] = {}
        
        min_sow = None
        max_sow = None
        min_cross_year = -float("inf")  # Change from inf to -inf
        max_cross_year = float("inf")
        
        for sow in unique_sows:
            cash_flow = results[sow]["cash_flow"]
            years = results[sow]["years"]
            
            cross_year = next((years[i] for i in range(len(cash_flow)) if cash_flow[i] >= 0), float("inf"))
            zero_crossing_years[scenario][sow] = cross_year
            
            if cross_year < max_cross_year:
                max_cross_year = cross_year
                max_sow = sow
            if cross_year > min_cross_year:
                min_cross_year = cross_year
                min_sow = sow
        if min_cross_year == max_cross_year:
            cash_flow_final_maxold = 0
            cash_flow_final_minold = 100000;
            final_year = all_years[-1]
            
            for sow in unique_sows:
                cash_flow_final_max = results[sow]["cash_flow"][-1]
                if cash_flow_final_max > cash_flow_final_maxold:
                    cash_flow_final_maxold = cash_flow_final_max
                    max_sow = sow
                if cash_flow_final_max < cash_flow_final_minold:
                    cash_flow_final_minold = cash_flow_final_max
                    min_sow = sow  
            #HERE!!!\n            
            # New logic: 
            #Check the final cumulative cash flow for all SOWs and select the SOW with the largest final value as max, and smallest as min.

        # (After computing avg_costs, std_costs, and stacked_avg_costs)
        zero_crossing_years[scenario] = {}
        if min_sow is not None and max_sow is not None:
            min_cash_flow = results[min_sow]["cash_flow"]
            max_cash_flow = results[max_sow]["cash_flow"]
        else:
            raise ValueError("No valid SOWs found for min or max cumulative cash flow crossing.")

        representative_sow = unique_sows[0]
        income_years = results[representative_sow]["years"]
        income_values = results[representative_sow]["income"]

        if scenario == 'timeseries_stoch_stochastic':
            Scenname = 'Stochastic'
        else:
            Scenname = 'Spines'

        
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.4
        x_positions = np.arange(len(all_years))
        #print(stacked_avg_costs)
        # Stacked bars for costs
        cost_labels = ["Cost_Inv","Cost_Act", "Cost_Fom", "Cost_Comx", "Cost_Flo"]
        cost_names =  ["Annuity loan","Var OPEX", "Fix OPEX", "CO2 tax", "Import"]
        cost_colour = ["purple","lightblue", "deepskyblue", "black", "khaki"]
        stacked_bottom = np.zeros(len(all_years))
        for cost_label, name, colour in zip(cost_labels,cost_names,cost_colour):
            if cost_label in stacked_avg_costs:
                ax.bar(x_positions, stacked_avg_costs[cost_label], width=bar_width, bottom=stacked_bottom, label=name, color=colour)
                stacked_bottom += stacked_avg_costs[cost_label]
                #print(stacked_avg_costs[cost_label])
        #yerr=std_costs
        percentage_contributions = {}

        # Calculate the percentage for each cost component
        for cost_label in cost_labels:
            if cost_label in stacked_avg_costs:
                values = np.array(stacked_avg_costs[cost_label]).flatten()  # Ensure 1D array
                # Avoid division by zero
                percentages = (values / stacked_bottom) * 100  # Calculate percentage
                percentage_contributions[cost_label] = percentages
        
        # Dictionary to store the averaged percentage contributions for each cost_label
        averaged_percentage_contributions = {}

        for cost_label in percentage_contributions:
            # Flatten to 1D array to avoid shape issues
            contribution_values = np.array(percentage_contributions[cost_label]).flatten()

            # Step 1: Average the first three values
            avg_first_three = np.mean(contribution_values[:3])

            # Step 2: Group the remaining values in chunks of 5 and average
            avg_remaining = []
            for i in range(3, len(contribution_values), 5):
                chunk = contribution_values[i:i + 5]  # Take a chunk of 5 values
                avg_remaining.append(np.mean(chunk))

            # Ensure exactly 4 values: trim or pad with NaN as needed
            avg_values = [avg_first_three] + avg_remaining
            avg_values = avg_values[:4]  # Ensure only 4 values
            while len(avg_values) < 4:
                avg_values.append(np.nan)  # Pad with NaN if not enough values

            # Store the averaged percentage contribution values
            averaged_percentage_contributions[cost_label] = avg_values

        #Uncomment to output the averaged results
        #for cost_label, avg_values in averaged_percentage_contributions.items():
        #    print(f"{Scenname}: Averaged percentage contributions for {cost_label}: {avg_values}")
        
        # Income bar, activate if cashflow
        ax.bar(x_positions, income_values, width=bar_width, color="green", alpha=0.7, label="Income")
        
        # Cumulative cash flow, activate if cashflow
        ax.plot(x_positions, max_cash_flow, label="Max Cash Flow", color="darkorange", linestyle="dashed", marker="^")
        ax.plot(x_positions, min_cash_flow, label="Min Cash Flow", color="darkorange", linestyle="dotted", marker="v")

        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f"Cashflow for {Scenname} model")
        ax.set_xlabel("Year", fontsize=M)
        ax.set_ylabel("MEuro", fontsize=M)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(all_years, rotation=45, fontsize=M)
        plt.yticks(fontsize=M)
        
        #Activate if cashflow
        #ax.legend(loc='upper left',ncol=4) #loc='upper left',
        #If only costs
        #ax.legend(loc='best',ncol=5)
        #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        
        #  ncol=3, fancybox=True, shadow=True)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=M)
        #plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center',ncol=4)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        #Activate if cashflow
        #plt.savefig(f'{run}_CashFlow_{Scenname}.png', bbox_inches='tight')
        #For only costs
        #plt.savefig(f'{run}_Costs_{Scenname}.png', bbox_inches='tight')
        
        #Uncommented the savefunction, but it functions.
        #plt.savefig(f'{run}_CashFlowCost_{Scenname}.png',bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(10, 6))
        for sow in unique_sows:
            plt.plot(results[sow]["years"], results[sow]["cash_flow"], label=f"SOW {int(sow)}", alpha=0.5)

        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        plt.title(f"Cashflow trajectories for {Scenname}")
        plt.xlabel("Year",fontsize=M)
        plt.ylabel("MEuro",fontsize=M)
        plt.xticks(rotation=45,fontsize=M)
        plt.yticks(fontsize=M)
        plt.xlim(2030,2047)
        plt.legend(loc="upper left", fontsize=M, ncol=2)
        #ax.legend()
        #plt.legend()
        plt.grid()
        
        #Removed the savefig
        #plt.savefig(f'{run}_Cash_Flow_Trajectories_{Scenname}.png', bbox_inches='tight')
        plt.show()
        
        std_dev_results[scenario]=std_costs 

        std_dev_results[scenario] = std_costs
        total_results[scenario] = results
    return [std_dev_results, npv_results, irr_results, zero_crossing_years], [total_results]


#%% Second function 
def Cash_flow_bar_stoch_avg2(file_path, run):

    data = pd.read_csv(file_path, delimiter=';', decimal=",")
    data['Period'] = pd.to_numeric(data['Period'], errors='coerce')  
    data['Vintage'] = pd.to_numeric(data['Vintage'], errors='coerce')
    data['Pv'] = pd.to_numeric(data['Pv'], errors='coerce')
    data['Sow'] = pd.to_numeric(data['Sow'], errors='coerce')  

    scenarios = ['timeseries_stoch_stochastic', 'timeseries_stoch_spines']
    std_dev_results = {}
    npv_results = {}
    irr_results = {}
    zero_crossing_years = {}
    total_results = {}
    for scenario in scenarios:
        scenario_data = data[data['Scenario'] == scenario]
        
        unique_sows = scenario_data['Sow'].dropna().unique()
        results = {}

        period_to_years = {
            2025: range(2022, 2028),
            2030: range(2028, 2033),
            2035: range(2033, 2038),
            2040: range(2038, 2043),
            2045: range(2043, 2048)
        }

        for sow in unique_sows:
            sow_data = scenario_data[scenario_data['Sow'] == sow]
            
            costs = {}
            income = {}
            cumulative_cash_flow = {}

            for _, row in sow_data.iterrows():
                attribute = row['Attribute']
                process = row['Process']
                period = row['Period']
                vintage = row['Vintage']
                value = row['Pv']

                if attribute in ['Cost_Act', 'Cost_Comx', 'Cost_Fom', 'Cost_Flo']:
                    if attribute == 'Cost_Flo' and process == 'EXPELC-DKW':
                        #CHANGE HERE, IT WAS if period == 2030
                        #yearly_value = (-value) / (3 if period == 2030  else len(period_to_years[period]))
                        yearly_value = (-value) #/ (3 if period in (2030, 2022, 2025)  else len(period_to_years[period]))
                        for year in (range(2030, 2033) if period in (2030,2022,2025) else period_to_years[period]):
                        #for year in (range(2030, 2033) if period == 2030 else period_to_years[period]):
                            income[year] = income.get(year, 0) + yearly_value
                    else:
                        yearly_value = (-value) #/ (3 if period == 2030 else len(period_to_years[period]))
                        for year in (range(2030, 2033) if period == 2030 else period_to_years[period]):
                            costs[year] = costs.get(year, 0) + yearly_value
                elif attribute == 'Cost_Inv':
                    yearly_value = (-value) #/ (3 if period == 2030 else len(period_to_years[period]))
                    for year in (range(2030, 2033) if period == 2030 else period_to_years[period]):
                        costs[year] = costs.get(year, 0) + yearly_value

            all_years = sorted(set(costs.keys()).union(set(income.keys())))
            yearly_costs, yearly_income, cumulative_cash_flow_values, cash_flow_normal = [], [], [], []
            total_cash_flow = 0

            for year in all_years:
                cost = costs.get(year, 0)
                income_value = income.get(year, 0)
                cash_flow = income_value + cost
                total_cash_flow += cash_flow

                yearly_costs.append(cost)
                yearly_income.append(income_value)
                cumulative_cash_flow_values.append(total_cash_flow)
                cash_flow_normal.append(cash_flow)
            results[sow] = {
                "years": all_years,
                "costs": yearly_costs,
                "income": yearly_income,
                "cash_flow": cumulative_cash_flow_values,
                "cash_flow2" : cash_flow_normal
            }
        
        npv_values = {sow: sum(results[sow]["cash_flow2"]) for sow in unique_sows}
        irr_values = {sow: npf.irr(results[sow]["cash_flow2"]) for sow in unique_sows}
        #
        npv_results[scenario] = npv_values
        irr_results[scenario] = irr_values
        
        all_years = sorted(set().union(*(res["years"] for res in results.values())))
        avg_costs, min_cash_flow, max_cash_flow, std_costs = [], [], [], []
        

        for year in all_years:
            year_costs = []
            for sow in unique_sows:
                if year in results[sow]["years"]:  # Ensure year exists for SOW
                    year_index = results[sow]["years"].index(year)
                    year_costs.append(results[sow]["costs"][year_index])
                else:
                    year_costs.append(0)  # Append zero if no cost entry

            avg_costs.append(np.mean(year_costs))
            std_costs.append(np.std(year_costs))

        zero_crossing_years[scenario] = {}
        
        min_sow = None
        max_sow = None
        min_cross_year = -float("inf")  # Change from inf to -inf
        max_cross_year = float("inf")

        for sow in unique_sows:
            cash_flow = results[sow]["cash_flow"]
            years = results[sow]["years"]

            cross_year = next((years[i] for i in range(len(cash_flow)) if cash_flow[i] >= 0), float("inf"))
            zero_crossing_years[scenario][sow] = cross_year

            if cross_year < max_cross_year:
                max_cross_year = cross_year
                max_sow = sow
            if cross_year > min_cross_year:
                min_cross_year = cross_year
                min_sow = sow
        #if scenario=='timeseries_stoch_stochastic':
        #    print('min 1', min_sow)
        #    print('max 1', max_sow)
        if min_cross_year == max_cross_year:
            #min_sow = None
            #max_sow = None
            #min_cross_year = -float("inf")  # Change from inf to -inf
            #max_cross_year = float("inf")
            cash_flow_final_maxold = 0
            cash_flow_final_minold = 100000;
            final_year = all_years[-1]
            
            for sow in unique_sows:
                cash_flow_final_max = results[sow]["cash_flow"][-1]
                
                #if scenario=='timeseries_stoch_stochastic':
                    #print('Sow', sow)
                    
                    #print('This is the new max value',cash_flow_final_max)
                    #print('This is the new min value',cash_flow_final_max)
                    #print('This is the old value',cash_flow_final_maxold)
                    #print('This is the old value',cash_flow_final_minold)
                #years = results[sow]["years"]
                if cash_flow_final_max > cash_flow_final_maxold:
                    cash_flow_final_maxold = cash_flow_final_max
                    max_sow = sow
                    #if scenario=='timeseries_stoch_stochastic':
                    #    print('Hello, this is the max',max_sow )
                    #print(max_sow)
                if cash_flow_final_max < cash_flow_final_minold:
                    cash_flow_final_minold = cash_flow_final_max
                    min_sow = sow  
                    #print(min_sow)
                
        #print(scenario)
        # Ensure min_sow and max_sow are not None before accessing them, since that happened a few times
        if min_sow is not None and max_sow is not None:
            min_cash_flow = results[min_sow]["cash_flow"]
            max_cash_flow = results[max_sow]["cash_flow"]
        else:
            
            raise ValueError("No SOWs found for min or max crossing.")
        
        representative_sow = unique_sows[0]
        income_years = results[representative_sow]["years"]
        income_values = results[representative_sow]["income"]
        
        if scenario == 'timeseries_stoch_stochastic':
            Scenname = 'Stochastic'
        else:
            Scenname = 'Spines'
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.4
        x_positions = np.arange(len(all_years))
        
        ax.bar(x_positions, avg_costs, yerr=std_costs, width=bar_width, color="red", alpha=0.7, label="Avg Costs")
        ax.bar(x_positions, income_values, width=bar_width, color="green", alpha=0.7, label="Income")

        ax.plot(x_positions, max_cash_flow, label="Max Cumulative Cash Flow", color="darkorange", linestyle="dashed", marker="o")
        ax.plot(x_positions, min_cash_flow, label="Min Cumulative Cash Flow", color="darkorange", linestyle="dotted", marker="o")
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f"Cashflow trajectories for {Scenname}")
        ax.set_xlabel("Year",fontsize=M)
        ax.set_ylabel("MEuro", fontsize=M)
        ax.set_xticks(x_positions)
        ax.set_ylim()
        ax.set_xticklabels(all_years, rotation=45,fontsize=M)
        plt.yticks(fontsize=M)
        #ax.legend(fontsize=L)
        #ax.legend()
        ax.legend(loc='upper left', ncol=4)
        #plt.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        #Uncomment if you wish to save graph as an image
        #plt.savefig(f'{run}_CashFlow_{Scenname}', bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(10, 6))
        for sow in unique_sows:
            plt.plot(results[sow]["years"], results[sow]["cash_flow"], label=f"SOW {int(sow)}", alpha=0.5)

        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        plt.title(f"Cashflow trajectories for {Scenname}")
        plt.xlabel("Year",fontsize=M)
        plt.ylabel("MEuro",fontsize=M)
        plt.xticks(rotation=45,fontsize=M)
        plt.yticks(fontsize=M)
        plt.legend(loc="upper left", fontsize=M, ncol=2)
        #ax.legend()
        #plt.legend()
        plt.grid()
        #plt.savefig(f'{run}_Cash_Flow_Trajectories_{Scenname}.png', bbox_inches='tight')
        plt.show()
        std_dev_results[scenario]=std_costs 
        
        total_results[scenario] = results
    return [std_dev_results, npv_results, irr_results, zero_crossing_years], [total_results]

#%%Third function, specifically made for deterministic scenario

def Cash_flow_bar_det(file_path, run):
    # Load the dataset
    data = pd.read_csv(file_path, delimiter=';', decimal=",")

    # Convert Period and Vintage again
    data['Period'] = pd.to_numeric(data['Period'], errors='coerce')  
    data['Vintage'] = pd.to_numeric(data['Vintage'], errors='coerce')
    data['Pv'] = pd.to_numeric(data['Pv'], errors='coerce')

    # Filter deterministic scenario
    deterministic_data = data[data['Scenario'] == 'deterministic']

    # Initialize dictionaries
    costs = {}
    income = {}

    # Define milestone years, this one could be remade into an input to the functions, in case milestone years change
    period_to_years = {
        2025: range(2023, 2028),
        2030: range(2028, 2033),
        2035: range(2033, 2038),
        2040: range(2038, 2043),
        2045: range(2043, 2048)
        }
    cost_components = {attr: {} for attr in ['Cost_Act', 'Cost_Comx', 'Cost_Fom', 'Cost_Flo','Cost_Inv' ]}
    # Iterate over rows to calculate costs and incomes
    for _, row in deterministic_data.iterrows():
        attribute = row['Attribute']
        process = row['Process']
        period = row['Period']
        #vintage = row['Vintage']
        value = row['Pv']
        
        if attribute in ['Cost_Act', 'Cost_Comx', 'Cost_Fom', 'Cost_Flo']:
            if attribute == 'Cost_Flo' and process == 'EXPELC-DKW':
                yearly_value = (value * -1) #/ (3 if period == 2030 else len(period_to_years[period]))
                for year in (range(2030, 2033) if period == 2030 else period_to_years[period]):
                    income[year] = income.get(year, 0) + yearly_value
            elif attribute == 'Cost_Flo' and process in ['IMPELC-DKW', 'MINGAS1']:
                yearly_value = (value * -1) #/ (3 if period == 2030 else len(period_to_years[period]))
                for year in (range(2030, 2033) if period == 2030 else period_to_years[period]):
                    costs[year] = costs.get(year, 0) + yearly_value
                    cost_components[attribute][year] = cost_components[attribute].get(year, 0) + yearly_value
            else:
                yearly_value = (value * -1) #/ (3 if period == 2030 else len(period_to_years[period]))
                for year in (range(2030, 2033) if period == 2030 else period_to_years[period]):
                    costs[year] = costs.get(year, 0) + yearly_value
                    cost_components[attribute][year] = cost_components[attribute].get(year, 0) + yearly_value
        elif attribute == 'Cost_Inv':
            yearly_value = (-value) #/ (3 if period == 2030 else len(period_to_years[period]))
            for year in (range(2030, 2033) if period == 2030 else period_to_years[period]):
                costs[year] = costs.get(year, 0) + yearly_value
                cost_components[attribute][year] = cost_components[attribute].get(year, 0) + yearly_value
    
    #removed notes here, check your backup note section
    
    # Generate sorted years and initialize lists
    all_years = sorted(set(costs.keys()).union(set(income.keys())))
    yearly_costs = []
    yearly_income = []
    cumulative_cash_flow_values = []
    total_cash_flow = 0
    yearly_cash_flows = []

    # Calculate yearly and cumulative cash flow
    for year in all_years:
        cost = costs.get(year, 0)
        income_value = income.get(year, 0)
        cash_flow = income_value + cost
        total_cash_flow += cash_flow

        yearly_cash_flows.append(cash_flow)
        yearly_costs.append(cost)
        yearly_income.append(income_value)
        cumulative_cash_flow_values.append(total_cash_flow)
    results = {
        "years": all_years,
        "costs": yearly_costs,
        "income": yearly_income,
        "cash_flow": cumulative_cash_flow_values,
        "cost_components": cost_components
    }
    # **Calculate NPV (Sum of discounted cash flows)**
    npv = sum(cumulative_cash_flow_values)

    # **Calculate IRR**
    try:
        irr = npf.irr([yearly_cash_flows[0]] + yearly_cash_flows[1:])  # First year investment + rest
    except:
        irr = None  # IRR cannot be computed if no positive cash flows exist
    #print(yearly_cash_flows)
    # **Find break-even year (first positive cumulative cash flow)**
    break_even_year = next((all_years[i] for i, val in enumerate(cumulative_cash_flow_values) if val > 0), None)


    stacked_avg_costs = {attr: [] for attr in ['Cost_Act', 'Cost_Comx', 'Cost_Fom', 'Cost_Flo','Cost_Inv']}
    for year in all_years:
        for attr in ['Cost_Act', 'Cost_Comx', 'Cost_Fom', 'Cost_Flo','Cost_Inv']:
            attr_values = []
            if year in results['cost_components'][attr]:
                attr_values.append(results['cost_components'][attr][year])
            else:
                attr_values.append(0)
            stacked_avg_costs[attr].append(attr_values)

    # **Plot results**
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.4
    x_positions = np.arange(len(all_years))


    cost_labels = ["Cost_Inv","Cost_Act", "Cost_Fom", "Cost_Comx", "Cost_Flo"]
    cost_names =  ["Annuity loan","Var OPEX", "Fix OPEX", "CO2 tax", "Import"]
    cost_colour = ["purple","lightblue", "deepskyblue", "black", "khaki"]
    stacked_bottom = np.zeros(len(all_years))
    for cost_label, name, colour in zip(cost_labels,cost_names,cost_colour):
        if cost_label in stacked_avg_costs:
            values = np.array(stacked_avg_costs[cost_label]).flatten()  # Convert to 1D array
            if values.shape[0] == stacked_bottom.shape[0]:  # Ensure correct shape
                ax.bar(x_positions, values, width=bar_width, bottom=stacked_bottom, label=name, color=colour)
                stacked_bottom += values  # Update stacked values correctly
            else:
                raise ValueError(f"Shape mismatch: {values.shape} vs {stacked_bottom.shape}")
            #values = np.array(stacked_avg_costs[cost_label]).flatten()
            #ax.bar(x_positions, values, width=bar_width, bottom=stacked_bottom, label=name, color=colour)     
            #ax.bar(x_positions, stacked_avg_costs[cost_label], width=bar_width, bottom=stacked_bottom, label=name, color=colour)
            #stacked_bottom += stacked_avg_costs[cost_label]
    percentage_contributions = {}

    # Calculate the percentage for each cost component
    for cost_label in cost_labels:
        if cost_label in stacked_avg_costs:
            values = np.array(stacked_avg_costs[cost_label]).flatten()  # Ensure 1D array
            # Avoid division by zero
            percentages = (values / stacked_bottom) * 100  # Calculate percentage
            percentage_contributions[cost_label] = percentages

    # Output or use percentage_contributions as needed
    #for cost_label, percentages in percentage_contributions.items():
    #    print(f"Percentage contribution for {cost_label}: {percentages}")
    # Dictionary to store the averaged percentage contributions for each cost_label
    averaged_percentage_contributions = {}

    for cost_label in percentage_contributions:
        # Flatten to 1D array to avoid shape issues
        contribution_values = np.array(percentage_contributions[cost_label]).flatten()

        # Step 1: Average the first three values
        avg_first_three = np.mean(contribution_values[:3])

        # Step 2: Group the remaining values in chunks of 5 and average
        avg_remaining = []
        for i in range(3, len(contribution_values), 5):
            chunk = contribution_values[i:i + 5]  # Take a chunk of 5 values
            avg_remaining.append(np.mean(chunk))

        # Ensure exactly 4 values: trim or pad with NaN as needed
        avg_values = [avg_first_three] + avg_remaining
        avg_values = avg_values[:4]  # Ensure only 4 values
        while len(avg_values) < 4:
            avg_values.append(np.nan)  # Pad with NaN if not enough values

        # Store the averaged percentage contribution values
        averaged_percentage_contributions[cost_label] = avg_values

    # Output the averaged results
    for cost_label, avg_values in averaged_percentage_contributions.items():
        print(f"Det: Averaged percentage contributions for {cost_label}: {avg_values}")
    
    print('Yearly income', yearly_income)
    #ax.bar(x_positions, yearly_costs, width=bar_width, color="red", alpha=0.7, label="Avg Costs")
    ax.bar(x_positions, yearly_income, width=bar_width, color="green", alpha=0.7, label="Income")
    ax.plot(x_positions, cumulative_cash_flow_values, label="Cash Flow", color="darkorange", linestyle="dashed", marker="^")

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(f"Cashflow for Deterministic model")
    ax.set_xlabel("Year",fontsize=M)
    ax.set_ylabel("MEuro",fontsize=M)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_years, rotation=45,fontsize=M)
    plt.yticks(fontsize=M)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=M)
    #ax.legend(loc='upper left', ncol=3)
    #ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    #plt.savefig(f'{run}_CashFlow_deterministic', bbox_inches='tight')
    plt.savefig(f'{run}_CashFlowCost_deterministic', bbox_inches='tight')
    plt.show()

    # **Print NPV, IRR, and Break-even Year**
    #print(f"NPV: {npv:.2f} MEuro")
    #print(f"IRR: {irr * 100:.2f}%") if irr is not None else print("IRR: Not computable")
    #print(f"Break-even Year: {break_even_year}")



    return results #npv, irr, break_even_year, averaged_percentage_contributions


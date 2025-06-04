"""
This code was written by Lucas Moshoej as a part of the Master thesis: Investment Optimization of an Energy Capacity Portfolio using Stochastic Modelling

Cost plots (CAPEX, FOM, VAROM)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

S = 10
M = 12
L = 14

def costplot(Cost_data, scenarioChosen):
    
    #This part provides an overview of the cost and income data, and is used for plots further down
    cost_attributes = ["Cost_Act", "Cost_Fom", "Cost_Inv","Cost_Comx"]
    #costcheck = ['IMPDEMZ'] #checking for dummy values
    
    #This line below is needed in the if else statement further down, to insure the income is not called into the cost
    cost_attributes2 = ['Cost_Act', 'Cost_Inv', 'Cost_Fom','Cost_Comx', 'Cost_Flo']

    # Get the cost data, this one filters out the specific scenario wanted, along with the attributes defined above and then adds the Cost_Flo of the mining operation of gas as well. 
    costs_data = Cost_data[Cost_data["Scenario"].isin(scenarioChosen) & (Cost_data["Attribute"].isin(cost_attributes)) | Cost_data["Scenario"].isin(scenarioChosen) & ((Cost_data["Attribute"] == "Cost_Flo") & (Cost_data["Process"] == "MINGAS1"))]

    # #Get the income, which is just the export of electricity
    income_data = Cost_data[Cost_data["Scenario"].isin(scenarioChosen) & (Cost_data["Attribute"] == "Cost_Flo") & (Cost_data["Process"] == "EXPELC-DKW")] 
    
    
    costs_data['Period'] = costs_data['Period'].astype(int)
    income_data['Period'] = income_data['Period'].astype(int)

    milestone_years = sorted(costs_data['Period'].dropna().astype(int).unique())
    
    #The deterministic model does not have any SOWS and it was a bit messy with the plots. So a rigid solution was chosen. 
    if scenarioChosen == ['deterministic']:
        
        #Loop over milestone years
        for year in milestone_years:
            
            #
            year_costs = costs_data[costs_data["Period"] == year]
            year_income = income_data[income_data["Period"] == year]

            

            total_cost = sum(year_costs['Pv'])
          
            income_value = -year_income['Pv'].sum() #simply to handle the dataframe, there is only the cost_flo income, so this simply returns it as the float I need
            profit_value = income_value - total_cost

            # Prepare the bar plot
            labels = ['Cost', 'Income', 'Profit']
            x = np.arange(len(labels))
            width = 0.5  # Bar width

            fig, ax = plt.subplots(figsize=(10, 6))
            cost_colors = {'Cost_Act': 'blue', 'Cost_Inv': 'orange', 'Cost_Fom': 'purple', 'Cost_Flo': 'red', 'Cost_Comx':'black'}
            # Stacked bar for costs
            bottom = 0  # Start stacking from zero
            
            #I have to redefine the cost_attributes that I had from before, 
            #as it does not include cost_flo, as that would also save the cost_flo export value, which is the income
            
            for cost, attr in zip(year_costs['Pv'], cost_attributes2):
               
                ax.bar(x[0], cost, width, label=attr, bottom=bottom, color=cost_colors[attr])
                bottom += cost  # Update bottom for stacking
            
            # Separate bars for income and profit
            ax.bar(x[1], income_value, width, label='Income', color='green', alpha=0.7)
            ax.bar(x[2], profit_value, width, label='Profit', color='c', alpha=0.7)

            # Configure the plot
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('MEuro')
            ax.set_title(f'Cost, Income, and Profit for Period {year} (Deterministic)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend(title="Cost Components and Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

            #Print the values as a table
            #print(f"\nYear: {year}")
            #print(f"{'Category':<15} {'Value (MEuro)':>15}")
            #print(f"{'-'*35}")
            #for comp, value in zip(cost_attributes2, year_costs['Pv']):
            #    print(f"{comp:<15} {value:>15.2f}")
            #print(f"{'Total Cost':<15} {total_cost:>15.2f}")
            #print(f"{'Income':<15} {income_value:>15.2f}")
            #print(f"{'Profit':<15} {profit_value:>15.2f}")
            #print("\n")
    else:  #Stoch
        scenname=scenarioChosen[0][17:]
        # Loop through milestone years to generate plots
        #For later use:
        
        scatter_data = []
        unique_sows = sorted(costs_data["Sow"].unique())

        for year in milestone_years:
            if year == 2025:
                continue
            #Get the current yearly data
            year_costs = costs_data[costs_data["Period"] == year]
            year_income = income_data[income_data["Period"] == year]
            if year_income.empty:  # Due to persistent and odd errors
                print(f"No income data found for Period {year}")

            # redefine attributes for costs
            #just use cost_attributes2 now instead of:
            #attributes = ['Cost_Act', 'Cost_Inv', 'Cost_Fom', 'Cost_Flo', 'Cost_Comx']
            #sows = [str(i) for i in range(1, len(unique_sows))]  # Convert Sows to string values to 

            # Initialize dictionaries to store Pv values for costs and income
            cost_data2 = {attr: [] for attr in cost_attributes2}
            PVincome_data2 = []  # For income values
            profit_data = []  # (income-cost)
            # Loop through each SOW and calculate profit
            for sow in unique_sows:
                for attr in cost_attributes2:
                    row = year_costs[(year_costs['Sow'] == sow) & (year_costs['Attribute'] == attr)]
                    pv_value = row['Pv'].values[0] if not row.empty else 0
                    cost_data2[attr].append(pv_value)

                # Income data for SOW
                PVincome_row = year_income[year_income['Sow'] == sow]
                PVincome_value = PVincome_row['Pv'].values[0] if not PVincome_row.empty else 0
                PVincome_data2.append(PVincome_value*(-1))  # Use positive values for the inverted axis

            #Profit
            for sow, cost_values, income_value in zip(unique_sows, zip(*cost_data2.values()), PVincome_data2):
                total_cost = sum(cost_values)
                profit = income_value - total_cost   # Profit = Income - total cost
                profit_data.append(profit)

            #getting the profit data appended, along with the corresponding year
            scatter_data.append((str(year), profit_data))
            # (Below) Generate bar positions for SOWs because I need more space to make the graph eligble
            x = np.arange(len(unique_sows)) 
            width = 0.3 #width of bars

            fig, ax1 = plt.subplots(figsize=(14, 6))

            cost_colors = {'Cost_Act': 'blue', 'Cost_Inv': 'orange', 'Cost_Fom': 'purple', 'Cost_Flo': 'red', 'Cost_Comx':'black'}
            
            # Initialize bottom values for stacking, while keeping the colours.
            bottom = np.zeros(len(unique_sows))

            # stacked bars for costs
            for attr in cost_attributes2:
                ax1.bar(x, cost_data2[attr], width, label=attr, color=cost_colors[attr], bottom=bottom)
                bottom += np.array(cost_data2[attr])  # Update bottom for stacking

            # Plotting income bars (income multiplied by -1 earlier as they are negative values when output by VEDA)
            ax1.bar(x + width, np.array(PVincome_data2), width, label='Income', color='green')

            # Plotting profit bars
            ax1.bar(x + width * 2, profit_data, width, label='Profit', color='c')
            
            ax1.set_xlabel('State of the World (Sow)')
            ax1.set_ylabel('Costs & Income (MEuro)', color='black')
            ax1.set_xticks(x)
            ax1.set_xticklabels(unique_sows)
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)

            # Moving the legend to the top,
            ax1.legend(loc='upper center', bbox_to_anchor=(1.05, 1), ncol=1)
            # I have to Adjust the plot to leave space for the legend cause it was all over the place
            plt.subplots_adjust(top=0.8)  # value just sorta figured out with trial and error. 

            plt.title(f'Cost and Income by SOW in {year}, {scenname} model')
            plt.tight_layout()
            plt.show()


        # Calculate variances
        variances_by_year = [np.var(profits, ddof=1) for _, profits in scatter_data]  # ddof=1 for sample variance, _, to ignore the years in the tuples

        # Display variances alongside the years
        #for year, variance in zip([year for year, _ in scatter_data], variances_by_year):
        #    print(f"Year {year}: Variance of profits = {variance:.2f}")

        fig, ax = plt.subplots(figsize=(10, 6))

        for (year, profits), variance in zip(scatter_data, variances_by_year): #this unpacks the tuple contained within scatter data and the variance contained within variances_by_year
            x_values = [year] * len(profits)
            ax.scatter(x_values, profits, label=f"{year} (Var: {variance:.2f})")
        
        if year == 2030: #Might want to change this later, it was just the stochastic one which had an issue in 2030 with this...
            location = -20
        else:
            location = 20
        # Shoow the variance near the top of each cluster of points - uncomment if needed
        #ax.text(
        #    year, max(profits) + location, f"Var: {variance:.2f}", 
        #    ha='center', va='bottom', fontsize=10, color='red'

        ax.set_title(f"Profit across SOWs, {scenname} model", fontsize=14)
        ax.set_xlabel("Year", fontsize=M)
        ax.set_ylabel("Profit (MEuro)", fontsize=M)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title="Year")

        plt.tight_layout()
        plt.show()    


        #variance annotations to the boxplot
        fig, ax = plt.subplots(figsize=(14, 6))
        profits_by_year = [profits for _, profits in scatter_data]

        ax.boxplot(profits_by_year, labels=[str(year) for year, _ in scatter_data], patch_artist=True)

        # Showing variances above the boxes
        for i, (year, variance) in enumerate(zip([year for year, _ in scatter_data], variances_by_year)):
            ax.text(i + 1, max(profits_by_year[i]) + 1, f"Var: {variance:.2f}", ha='center', fontsize=S, color='black')

        # Customizing the plot
        ax.set_xlabel('Milestone Year')
        ax.set_ylabel('Profit (MEuro)')
        ax.set_title(f'Profit Distribution and Variance Across Milestone Years and SOWs, {scenname} model')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        #visual options for the boxes
        for box in ax.artists:
            box.set_facecolor('lightblue')
            box.set_edgecolor('black')
            box.set_alpha(0.8)

        plt.tight_layout()
        plt.show()
        
        return variances_by_year, costs_data, income_data, milestone_years
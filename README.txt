The first part I want to highlight from this project is the Linux .sh file called vtrun, which calls my TIMES model to run on Linux without the intermediary program VEDA. This file is something I created based on a forum post I made, to which I also contributed.
Link to forum post: https://forum.kanors-emr.org/showthread.php?tid=1491

The project within this file are called through main.py, and require basic python libraries as detailed in requirements.txt. 
The code is focused on cashflow data analysis and visualization. 
The exact function I would like you to look at is in Cashflow_functions.py, called Cash_flow_bar_stoch_avg_with_stack.
Green text with 'PhD' highlights the relevant cell in main and the relevant function in Cashflow_functions.
Specific aspects of Cashflow_functions.py that I would highlight are: 
The modular structure with the dictionaries that are used to finally create stackable columns of costs data.
I simplify and direct complex and large model outputs from my model, so that they fit into graphs from which clearer interpretation is possible.

The rest of the code, and excel files, are there to ensure that the code does not stutter if all of main is run.
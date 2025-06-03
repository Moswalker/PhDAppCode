#!/bin/bash
### General options
### -- set the job Name --
#BSUB -J Spines_Scenario
### -- ask for number of cores (default: 1) --
#BSUB -n 5
### -- specify that the cores MUST BE on a single host! It's a SMP job! --
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm --
#BSUB -W 20:00 
### -- set the email address --
#BSUB -u s194813@student.dtu.dk
# -- request a machine with 4 GB of memory per process/core --
#BSUB -R "rusage[mem=4GB]"
# -- kill the job if ti exceeds 5 GB of memory per process/core --
#BSUB -M 5GB
### -- send notification at start -- NONE
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -e Spines_Scenario.err 
# set OMP_NUM_THREADS _and_ export! 
# OMP_NUM_THREADS=$LSB_DJOB_NUMPROC 
# export OMP_NUM_THREADS 
# Get paths to GAMS 47
export PATH=/appl/gams/47.6.0:$PATH
export LD_LIBRARY_PATH=/appl/gams/47.6.0:$LD_LIBRARY_PATH

### ------------------------------- Program_name_and_options
#set the casename
## Didn't work casename =$timeseries_stoch_spines

# set the path to times source file
times =$HOME/Stochrun/gams_srctimes.v4.8.1

# set the path to model data definition files
ddfiles=$HOME/Stochrun/gams_wrktimes

# execute gams to run the model
gams timeseries_stoch_spines.run idir1=/zhome/d6/f/146745/Stochrun/gams_srctimes.v4.8.1 ps=0 gdx=timeseries_stoch_spines filecase=4

##../gams_srctimes.v4.8.1 idir1

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
HRV analysis
Created: 03/2022 - 05/2022
Python v3.8.8
Author: F. Smits

Basic algorithm for heart rate variability (HRV) analysis. This script was 
written during a Technical Medicine year 2 internship (TM2). The purpose of the
algorithm is to analyze the HRV of ICU patients. Further improvements include:
    - Adding timestamps to the outputfile in order to be able to analyze
      circadian rhythms
    - Improving artefact detection

Make sure the following files are stored in the same folder before running:
    - HRV_main.py
    - HRV_preprocessing.py
    - HRV_calculations.py
    - HRV_batchmode.py
    - files_id.txt
    
Output variables: 
    - batch_df: list containing dataframes with timestamps, raw- and filtered
      ECG signals per included patient
    - batch_nni: list containing arrays with nni per included patient [ms]
    - batch_nni_first: list containing arrays with nni per included patient 
      [ms], before ectopic beat- and outlier removal
    - batch_rpeaks: list containing arrays with rpeak locations [s]
    - batch_rpeaks_first: list containing array with rpeak locations [s]-
      before ectopic beat- and outlier removal

Output file: the output file is stored in the same folder as the current file.
    - HRVparameters.csv: file containing all calculated HRV parameters per 
      included patient for the first hour of ECG recording. Rows = patients,
      columns = HRV parameters
    - per5min.csv: file containing mean calculcated HRV parameters per 5-minute
      segment per included patient for the first hour of ECG recording. Rows =
      patients, columns = HRV parameters
"""
#%% Required modules
import numpy as np
import HRV_batchmode as workflow
import HRV_evaluation as viseval
import HRV_window as window


#%% HRV calculations for specified time with a specified window
# Specify input variables
patient_ids = 'files_id_testpacing.txt'                                               # .txt file containing patient_IDs
sampfreq = 125                                                                  # Sample frequency in Hertz [Hz]
lead = 'II'                                                                     # ECG lead to analyze 
HRVwindow = 5                                                                   # Number of minutes you want to analyze the HRV
Startmin = 10                                                                   # Specify after how many minutes analysis must start
Durationmin = 10                                                                # Specify period of analysis in minutes
export = list()              

starttime, endtime = window.create_window(HRVwindow, Startmin, Durationmin)

# Calculate HRV parameters per specified window segments
for i in np.arange(0, len(starttime), 1):
    print(i)
    # HRV calculations for all patients specified in patient_ids
    batch_df, batch_nni_first, batch_rpeaks_first, batch_nni, batch_rpeaks, export_all = workflow.workflow_batch(patient_ids, 
                                                                                                                 sampfreq, lead, 
                                                                                                                 starttime[i], endtime[i], 
                                                                                                                 HRVwindow)
    
    export_all.insert(0, 'time [min]', starttime[i]/60)
    export.append(export_all)  

 
exp = export[0]
for i in np.arange(0, len(starttime)-1, 1):
    exp = exp.append(export[i+1]) 

exp['patientID'] = exp.index
exp = exp.sort_values(['patientID', 'time [min]'])                              # Sort by patient, by time
exp.to_csv('pacing.csv')                                                       # Export dataframe to .csv file

#%% Visual evaluation 
   
viseval.visual_evaluation_rpeaks(batch_df, batch_rpeaks)                        # Visual evaluation of R-peak detection             
viseval.visual_evaluation_nni(batch_nni, batch_rpeaks, batch_nni_first,         # Visual evaluation of NNI correction
                              batch_rpeaks_first)    

#%% Create histogram per HRV parameter, icluding mean and standard deviation
    
# Manually select .csv file of choice (stored in same folder)
result = viseval.hrv_distribution('per5minhourresults.csv')                                    # Create histograms                               

    





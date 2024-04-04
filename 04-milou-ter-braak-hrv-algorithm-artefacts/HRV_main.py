#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRV analysis
Created: 01/03/2022
Last update: 23/02/2022
Python v3.9.5 ('base': conda)
Adjusted: M. ter Braak

Basic algorithm for heart rate variability (HRV) analysis. This script was
written during a Technical Medicine year 2 internship (TM2). The purpose of the
algorithm is to analyze the HRV of ICU patients.

Make sure the following files are stored in the same folder before running:
    - HRV_main.py
    - HRV_preprocessing.py
    - HRV_detection.py
    - HRV_calculations.py
    - HRV_validation.py
    - files_id.txt

Output file:
    - patient_hrvdata_{pt}.csv file containing all calculated HRV parameters for consecutive
      time segments including timestamp. Note; if multiple

"""
# %% Required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import HRV modules
import HRV_detection
import HRV_preprocessing
import HRV_calculations
import HRV_visualisation
# %% Specify initial variables

pt_records_file = '2_waves.txt'                                          # .txt file containing patient_IDs
sampfreq = 62.4725*4                                                            # Sample frequency in Hertz [Hz]
lead = ['II']                                                                   # ECG lead to analyze

startmin = 10                                                                   # Specify after how many minutes analysis must start
windowmin = 5                                                           # Number of minutes you want to analyze the HRV in for each analytical window
segmentmin = 60                                                                 # Specify the length of each segment. Note 1 window will represent the entire segment
segmentamount = 1                                                              # Specify the amount of segments you want to analyse

lst_all = []                                                                    # Create empty dataframe for future storage of hrv_parameters
pt_timepoints = []                                                                # Create empty list for future storage of time-axis
# %% Preprocessing
'''
After defining the initial variables, preprocessing will first define the windows we must analyze and the patient ID's that are required.
Thereafter the data will be loaded
'''
starttimes, stoptimes = HRV_preprocessing.create_window(startmin, windowmin, segmentmin, segmentamount)     # Create lists with specified start- and stoptimes for analysis
pt_records = HRV_preprocessing.load_pts(pt_records_file)                              # From the patient records file, extract patient record names

for ind_pt, pt in enumerate(pt_records):
    print(f'---- Start calculations for patient: {ind_pt + 1} ----')
    for ind_time, (start, stop) in enumerate(zip(starttimes, stoptimes)):
        for start_sec in np.arange(start, start+60*(segmentmin-windowmin), windowmin*60):
            print(f'Calculate segment {ind_time+1}/{segmentamount}, trying starttime: {start_sec/60} min')
            stop_sec = start_sec + windowmin*60
            ecg_record, pt_path, w_record_name, database_name, record_dir, w_record_header = HRV_preprocessing.load_data(pt, sampfreq, start_sec, stop_sec, lead)

            ecg_df = HRV_preprocessing.ecg_dataframe(ecg_record, sampfreq)
            
            # (Artefact) Detection
            '''
            The R-peaks are detected and common artefacts, such as ectopic beats, are removed. 
            '''
            ecg_df, clipping = HRV_detection.clipping(ecg_df, sampfreq)
            pacing = HRV_detection.pacespike(ecg_df)
            ecg_df, r_peaks_first, ecg_filtered, ecg_analysis, valid_signal = HRV_detection.ecg_rpeak(ecg_df, sampfreq)

            if ecg_analysis == 1:
                print(f'try next time segment: no peak detection')
                continue
            r_peaks_first, noisepeak, noisepeak_val = HRV_detection.peakprominence(r_peaks_first, ecg_df, ecg_filtered, sampfreq)
            bas_shift, selection, nni_first, time = HRV_detection.movement_art(ecg_df, r_peaks_first, sampfreq)
            r_peaks, nni, ectopic_beat = HRV_detection.ecg_ectopic_removal(r_peaks_first, nni_first)
            valid_signal = HRV_detection.ecg_check(noisepeak, pacing, clipping, ecg_analysis, windowmin, sampfreq, nni, selection, r_peaks, noisepeak_val, ectopic_beat)

            r_peaks.to_csv(f'r_peaks_{ind_pt+1}.csv')  # Timepoint of the detected R-peaks 
            if valid_signal == False:
                continue
            pt_timepoints += [ecg_df.at[0,'TimeStamp']]
            break

        if valid_signal == False:
            raise Exception(f"There was no good signal for the current segment (#{ind_time + 1})")
    
        # R peak validation
        HRV_visualisation.visualisation(ecg_df, r_peaks, time, selection, ectopic_beat, noisepeak_val, windowmin, segmentmin)

        # Heart Rate Variability Calculations
        '''
        Stukje uitleg over dit gedeelte van de code
        '''
        hrv_parameters = HRV_calculations.hrv_calculations(nni, sampfreq, windowmin)
        lst_all, tuplekeys = HRV_calculations.to_dataframe(hrv_parameters, ind_time, lst_all) 

        print(f'Done with calculations of segment {ind_time + 1}/{segmentamount}')
        if ind_time == segmentamount - 1:
            df_hrv = pd.DataFrame(lst_all, columns=tuplekeys)
            df_hrv['TimeStamp'] = pt_timepoints[-segmentamount:]
            df_hrv.to_csv(f'patient_hrvdata_{ind_pt+1}.csv')
            print(f'Dataframe of patient {ind_pt+1} saved as .csv-file\n')
# %%
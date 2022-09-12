#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:18:22 2022

@author: floorsmits
"""

import pandas as pd
import numpy as np

# HRV toolbox
import pyhrvg

# Signals toolbox
import biosppy
import pyhrv.tools as tools

# import the WFDB(WaveFormDataBase) package
import wfdb

import datetime


# Import HRV modules
import HRV_preprocessing as preproc
import HRV_calculations as hrvcalc
import HRV_batchmode as workflow
import HRV_evaluation as viseval
import HRV_window as window

#%% Function

def ecg_rpeak(ecg_df, sampfreq):
    """
    Function that uses bioSPPY toolbox in order to filter the ECG signal and
    detect R-peaks. NN intervals are calculated as well.
    Reference: https://github.com/PIA-Group/BioSPPy/blob/5fbfb1de5917eef2609ce985307546d37aa0a0ca/biosppy/signals/ecg.py 
    
    INPUT:
        ecg_df: DataFrame with rows = time [s], columns = ECG data [mV]
        sampfreq: sampling frequency of recording [Hz]
        
    OUTPUT:
        dataframe: ecg_df with added column with filtered ECG signal [mV]
        r_peaks: array containing all detected R-peaks [s]
        nni: array containing all calculated NN intervals [ms]   
    """
    sampling_rate = sampfreq
    dataframe = ecg_df
    ecg_filtered, r_peaks = biosppy.signals.ecg.ecg(dataframe.ecg_signal, 125, 
                                                    show=True)[1:3]            # [1:3] To get filtered signal and R-peaks
    dataframe = dataframe.assign(ecg_filtered = ecg_filtered)                   # Add filtered ECG signal to dataframe
    r_peaks =  r_peaks * 1/sampling_rate                                        # Convert from index to time in seconds
    nni = tools.nn_intervals(r_peaks)                                           # Calculate NN intervals
      
    return dataframe, r_peaks, nni



#%% Code

patient_ids = 'files_id_test1.txt'                                              # .txt file containing patient_IDs
sampfreq = 125                                                                  # Sample frequency in Hertz [Hz]
lead = 'II'                                                                     # ECG lead to analyze 
HRVwindow = 0.5                                                                   # Number of minutes you want to analyze the HRV
Startmin = 10                                                                   # Specify after how many minutes analysis must start
Durationmin = 1                                                                # Specify period of analysis in minutes
export = list()              

starttime, endtime = window.create_window(HRVwindow, Startmin, Durationmin)

with open(patient_ids) as f:                                                     
        lines = [x.strip() for x in list(f) if x]

# For every patient ID calculate HRV parameters
for line in lines:
    print(line)
    ecg = preproc.load_data(line, sampfreq, starttime, endtime, lead)  # Load data
    ecg_df = preproc.ecg_dataframe(ecg, sampfreq)
    ecg_df, r_peaks, nni = preproc.ecg_rpeak(ecg_df, sampfreq)         # R_peak detection and nni calculation


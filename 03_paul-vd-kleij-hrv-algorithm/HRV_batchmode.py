# -*- coding: utf-8 -*-
"""
HRV batchmode
Created: 10/2021 - 02/2022
Python v3.8
Author: M. Verboom
"""

# Basic import
import pandas as pd
import numpy as np
import os
from pathlib import Path
from pytictoc import TicToc

# HRV toolbox
import pyhrv

# Import HRV modules
import HRV_preprocessing as preproc
import HRV_calculations as hrvcalc

#%% Batchmode 
def workflow_batch(pt_ids, sampfreq, lead, starttime, endtime, hrvwindow):
    """
    Function for the HRV calculation in batchmode (multiple patients).
    
    INPUT:
        patient_ids: .txt file of patient ID from MIMIC-III database
        sampfreq: sampling frequency of recording [Hz]
        lead: ECG lead for analysis: 'I', "II", "V"
        starttime: starting time of ECG analysis
        endtime: ending time of ECG analysis
    
    OUTPUT:
        batch_dataframes: list containing dataframes with raw ecg,
                          filtered ecg and time of all patients
        batch_rpeaks: list containing all locations of r-peaks for all 
                      patients [s]
        batch_rpeaks_first: list containing array with rpeak locations [s]
                            before ectopic beat- and outlier removal
        batch_nni: list containing arrays with nni per included patient [ms]
        batch_nni_first: list containing arrays with nni per included patient 
                         [ms], before ectopic beat- and outlier removal
        export_all: dataframe with rows = patients, columns = HRV parameters  
    """
    
    badsignal = 0.25*60*hrvwindow*sampfreq
    
    # Create empty lists for storage of dataframes
    indices = list()
    batch_dataframes = list()                                                   
    batch_rpeaks = list()
    batch_rpeaks_first = list()
    batch_nni = list()
    batch_nni_first = list()
    batch_all = list()
    timestamp = list()
    timeend = list()    

    # Read .txt file containing patient IDs
    cwd = os.getcwd()
    cwd_path = Path(cwd)
    for item in cwd_path.glob(f'**/{pt_ids}'):
        with open(item) as f:                                                     
            lines = [x.strip() for x in list(f) if x]

    # For every patient ID calculate HRV parameters
    for line in lines:
        print(f'patient ID: {line}')
        try:
            ecg = preproc.load_data(line, sampfreq, starttime, endtime, lead)  # Load data
            ecg_df = preproc.ecg_dataframe(ecg, sampfreq)                      # Preprocessing
            ecg_df, r_peaks, ecg_filtered, clipping, pacing, ecg_analysis = preproc.ecg_rpeak(ecg_df, sampfreq)         # R_peak detection and nni calculation
            
            if ecg_analysis == 1:
                print(f'ecg analysis is 1 so something went wrong') 
                continue
            r_peak_prom, nni, noisepeak = preproc.peakprominence(r_peaks, ecg_df, ecg_filtered, sampfreq)
            r_peaks_ect, nni_ect = preproc.ecg_ectopic_removal(r_peak_prom, nni)        # Ectopic beat removal
    
            if noisepeak >= 1:
                print(f'this is the noisepeak {noisepeak}')
            removed = len(clipping) + noisepeak
            print(f'there were {len(clipping)} instances of clipping')
            
            if pacing > 10: 
                print(f'Too many pacing spikes: {pacing}')
                continue
            else: 
                indices.append(line)
            if removed > badsignal:
                print(f'Not enough signal, removed: {removed}')
                if line in indices:
                    indices.remove(line)
                continue
            
            r_peaks_real, nni_real = preproc.ecg_correct_int(r_peaks_ect, nni_ect)
            print(f'this is the length of nni_real: {len(nni_real)}')
            if not nni_real:
                print('No useful intervals')
                if line in indices: 
                    indices.remove(line)
                continue
            hrv_td, hrv_fd, hrv_nl = hrvcalc.hrv_results(nni=nni_real,                   # HRV calculations for td: timedomain, fd: frequency domain, nl: nonlinear
                                                         sampfreq=sampfreq, HRVwindow = hrvwindow)
            hrv_all = pyhrv.utils.join_tuples(hrv_td, hrv_fd, hrv_nl)               # Join tuples of td, fd and nl
            
            # print(f'this is hrv_all: {hrv_all.size()}')
            end_df = len(ecg_df)
    
            # Store dataframes per patient in list
            batch_dataframes.append(ecg_df) 
            batch_rpeaks.append(r_peaks_real)
            batch_nni.append(nni_real)
            batch_all.append(hrv_all)
            batch_nni_first.append(nni)
            batch_rpeaks_first.append(r_peaks)
            timestamp.append(ecg_df['TimeStamp'].iloc[0])
            timeend.append(ecg_df['TimeStamp'].iloc[end_df-1])
        except:
            print('Something goes wrong')
    # Exportfile
    #print (f'this is Batch all: {batch_all}')
    try:
        tuplekeys_all = batch_all[0].keys()
        lst_all = []

        for i in range(1,len(batch_all)):
            lst = []
            for key in tuplekeys_all:
                lst += [batch_all[i][key]]
            lst_all.append(lst)

        export_all = pd.DataFrame(lst_all, index=indices, columns=tuplekeys_all)        # Create dataframe for exportfile
        export_all.insert(0, 'Start Time [h:m:s]', timestamp)
        export_all.insert(1, 'End Time Window [h:m:s]', timeend)
        export_all = export_all.drop(['dfa_alpha1_beats', 'dfa_alpha2_beats'], axis=1)
        #export_all.to_csv('HRVparameters.csv')                                         # Write calculated parameters to .csv file
    except:
        print('output file ging niet goed')
    return batch_dataframes, batch_nni_first, batch_rpeaks_first, batch_nni, batch_rpeaks, export_all

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

# HRV toolbox
import pyhrv

# Import HRV modules
import HRV_preprocessing as preproc
import HRV_calculations as hrvcalc

#%% Batchmode 
def workflow_batch(patient_ids, sampfreq, lead, starttime, endtime, HRVwindow):
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
    
    pt_ids = patient_ids
    sampling_rate = sampfreq
    lead = lead
    starttime = starttime
    endtime = endtime
    badsignal = 0.25*60*HRVwindow*sampling_rate
    
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
    with open(pt_ids) as f:                                                     
        lines = [x.strip() for x in list(f) if x]

    # For every patient ID calculate HRV parameters
    for line in lines:
        print(line)
        try:
            ecg = preproc.load_data(line, sampling_rate, starttime, endtime, lead)  # Load data
            ecg_df = preproc.ecg_dataframe(ecg, sampling_rate)                      # Preprocessing
            ecg_df, r_peaks, ecg_filtered, clipping, pacing, ecg_analysis = preproc.ecg_rpeak(ecg_df, sampling_rate)         # R_peak detection and nni calculation
            if ecg_analysis == 1: 
                continue
            r_peak_prom, nni, noisepeak = preproc.peakprominence(r_peaks, ecg_df, ecg_filtered, sampling_rate)
            r_peaks_ect, nni_ect = preproc.ecg_ectopic_removal(r_peak_prom, nni)        # Ectopic beat removal 
    
            if noisepeak >= 1:
                print(noisepeak)
            removed = len(clipping) + noisepeak
            print(len(clipping))
            
            if pacing > 10: 
                print('Too many pacing spikes')
                print(pacing)
                continue
            else: 
                indices.append(line)
            if removed > badsignal:
                print('Not enough signal')
                print(removed)
                if line in indices:
                    indices.remove(line)
                continue
            
            r_peaks_real, nni_real = preproc.ecg_correct_int(r_peaks_ect, nni_ect)
            
            if not nni_real:
                print('No useful intervals')
                if line in indices: 
                    indices.remove(line)
                continue
            hrv_td, hrv_fd, hrv_nl = hrvcalc.hrv_results(nni=nni_real,                   # HRV calculations for td: timedomain, fd: frequency domain, nl: nonlinear
                                                         sampfreq=sampling_rate, HRVwindow = HRVwindow)
            hrv_all = pyhrv.utils.join_tuples(hrv_td, hrv_fd, hrv_nl)               # Join tuples of td, fd and nl
            
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
    try:
        tuplekeys_all = batch_all[0].keys()                                         # Names of HRV parameters
        matrix_all = []
                
        for key in tuplekeys_all:                                                   
            matrix_all += [batch_all[0][key]]    
             
        matrix_all = np.array(matrix_all)                                           # Final matrix with HRV parameters of first patient
        
        for i in range(1,len(batch_all)):                                           # Per patient
            stack = []
            
            for key in tuplekeys_all:
                stack += [batch_all[i][key]]                                         
                
            stack = np.array(stack)
            matrix_all = np.vstack((matrix_all, stack))                             # Add HRV parameters for patient i to final matrix
                
        export_all = pd.DataFrame(matrix_all, index=indices, columns=tuplekeys_all)   # Create dataframe for exportfile
        export_all.insert(0, 'Start Time [h:m:s]', timestamp)
        export_all.insert(1, 'End Time Window [h:m:s]', timeend)
        export_all = export_all.drop(['dfa_alpha1_beats', 'dfa_alpha2_beats'], axis=1)
        #export_all.to_csv('HRVparameters.csv')                                      # Write calculated parameters to .csv file
    except:
        print('output file ging niet goed')
    return batch_dataframes, batch_nni_first, batch_rpeaks_first, batch_nni, batch_rpeaks, export_all

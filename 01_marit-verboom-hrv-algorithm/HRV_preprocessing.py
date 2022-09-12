# -*- coding: utf-8 -*-
"""
HRV analysis
Created: 10/2021 - 02/2022
Python v3.8
Author: M. Verboom
"""
#%%
# Required modules
# Basic import
import pandas as pd
import numpy as np


# Signals toolbox
import biosppy
import pyhrv.tools as tools

# import the WFDB(WaveFormDataBase) package
import wfdb

#%%
def load_data(patient_id, sampfreq, sampfrom, sampto, lead='II'):
    """
    Function to load data from MIMIC-III database.
    
    INPUT:
        patient_id: ID number of patient for analysis
        sampfreq: sampling frequency of recording [Hz]
        starttime: starting time of ECG analysis
        endtime: ending time of ECG analysis
        lead: ECG lead for analysis: 'I', "II", "V"
        
    OUTPUT:
        record: data from waveform database 
    """
    pt_id = patient_id
    sampling_rate = sampfreq
    Sampfrom = sampling_rate * sampfrom                                         # The starting sample number to read for all channels
    Sampto = sampling_rate * sampto                                             # The sample number at which to stop reading for all channels
    LeadWanted = [lead]                                                         # Lead that is used for the analysis
    record = wfdb.rdrecord(pt_id[-7:], sampfrom = Sampfrom, sampto = Sampto, 
                           pn_dir=('mimic3wdb/'+pt_id), channel_names =
                           LeadWanted)
    return record

def ecg_dataframe(record, sampfreq):
    """
    Function to create dataframe with timestamps and raw ECG data
    INPUT:
        record: data from waveform database
        sampfreq: sampling frequency of recording [Hz]
        
    OUTPUT:
        ecg_df: DataFrame with rows = time [s], columns = ECG data [mV]
    """   
    t = np.linspace(0, record.sig_len, record.sig_len)                          # Length of signal                      
    time = pd.DataFrame(t)                                                      # Create dataframe for plotting
    time.columns = ['Time']                                                     # Change name of column to 'Time'
    ecg_df = time.assign(ecg_signal = pd.DataFrame(record.p_signal))            # Add ECG signal to dataframe
    
    #NaN removal
    index = np.arange(0, ecg_df.ecg_signal.first_valid_index(), 1)
    ecg_df=ecg_df.drop(labels=index, axis=0)        
    ecg_df = ecg_df.fillna(0)
    ecg_df.Time = ecg_df.Time/sampfreq                                          # Time axis in seconds (125 Hz)
        
    return ecg_df
           

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
                                                    show=False)[1:3]            # [1:3] To get filtered signal and R-peaks
    dataframe = dataframe.assign(ecg_filtered = ecg_filtered)                   # Add filtered ECG signal to dataframe
    r_peaks =  r_peaks * 1/sampling_rate                                        # Convert from index to time in seconds
    nni = tools.nn_intervals(r_peaks)                                           # Calculate NN intervals
                
    return dataframe, r_peaks, nni

def ecg_ectopic_removal(r_peaks, nni):
    """
    Function for the removal of outliers and ectopic beats.
    
    INPUT:
        r_peaks: array containing all detected R-peaks [s]
        nni: array containing all calculated NN intervals [ms]
        
    OUTPUT:
        rrn_true: corrected array containing all R-peaks [s]
        nni_true: corrected array containing all NN intervals [ms] 
        
    """
    
    n = np.arange(1, len(nni), 1)[10:-2]                                        # Number of nni intervals from 10th until end-5th value 
    nni_true = nni.copy()
    rrn = r_peaks[:-1]
    rrn_true = rrn.copy()
    index = []
    
    for i in np.arange(1, len(nni), 1):
         if nni[i] > 6000 or nni[i] < 300:                                      # Delete NNI and corresponding R-peaks of non-physiological values
            index = index + [i]
    
    for i in n:                                                                 # For every nni
        if nni[i] < 0.75*(np.mean(nni[i-11:i-1])):                              # If nni < 0.75 times the mean of the previous 10 nnis
            if nni[i+1] < 0.75*(np.mean(nni[i-11:i-1])):                        # And if consecutive nni < 0.75 times the mean of the previous 10 nnis
                if nni[i+2] > 0.75*(np.mean(nni[i-11:i-1])):                    # And if consecutive nni is 'normal' again
                    nni_true[i] = np.median(nni[i-11:i-1])                      # True value of nni is median of previous 10 nnis
                    index = index + [i+1]                                       # Index of consecutive nni/rpeak (will be removed later on)
        if nni[i] < 0.85*(np.mean(nni[i-11:i-1])):                              # If nni < 0.75 times the mean of the previous 10 nnis        
            if nni[i+1] > 1.15*(np.mean(nni[i-11:i-1])):                        # And if conescutive nni >1.15 times the mean of the previous 10 nnis
                if nni[i+2] > 0.75 * (np.mean(nni[i-11:i-1])):                  # And is consecutive nni is 'normal' again
                    nni_true[i] = np.median(nni[i-11:i-1])                      # True value of nni is median of prevous 10 nnis     
                    rrn_true[i] = rrn_true[i-1] + nni_true[i]/1000              # True value of r peak is previous r-peak + nni
                    index = index + [i+1]                                       # Index of consecutive nni/rpeak (will be removed later on)        
        if nni[i] > 1.15*(np.mean(nni[i-11:i-1])) or nni[i] < 0.85*(np.mean(nni[i-11:i-1])):
            nni_true[i] = np.median(nni[i-11:i-1])
    
    nni_true = np.delete(nni_true, index)                                       # Remove all 'extra' nni intervals from data
    rrn_true = np.delete(rrn_true, index)                                       # Remove all 'extra' r peaks from data
    
    return rrn_true, nni_true
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Artefact) Detection
Created: 10/3/2022 - 05/2022
Last update: 07/11/2022
Python v3.9.5('base': conda)
Author: P.A. van der Kleij
"""
import numpy as np
import pandas as pd
from math import ceil
import biosppy
import scipy
from pyhrv import tools

#%% Functions
def clipping(ecg_df, sampfreq):
    '''
    Function to determine whether or not there are consecutive values that are 
    exactly the same. These values and their surrounding get adapted to the 
    median value of the signal. 

    Parameters
    ----------
    ecg_df : Dataframe
        Dataframe that contains all the data around the ECG.
    sampfreq : Integer
        Sample frequency of the data.

    Returns
    -------
    ecg_df : Dataframe
        ECG dataframe with added column of same value.
    clippingdf2 : Dataframe
        Contains only all the changed parameters.

    '''
    max_ecg = max(ecg_df['ecg_signal'])
    min_ecg = min(ecg_df['ecg_signal'])
    ecg_minmax = ecg_df[(ecg_df['ecg_signal'] == max_ecg) | (ecg_df['ecg_signal'] == min_ecg)] #first only take values that contain max and min, since clipping can only occur there.
    
    labels = ((ecg_minmax['index'] - ecg_minmax['index'].shift() > 1) | (ecg_minmax['ecg_signal'] != ecg_minmax['ecg_signal'].shift())).cumsum()
        # For the above line. It makes sure that if consequent indexes are found within the minmax signal they are labelled the same, since that could be clipping.
        # The first checks whether the indexes are differin by more than 1. The next checks whether they are both mins, or both maxes (since clipping cant have a min next to a max) 
    ecg_df['clipping'] = (labels.map(labels.value_counts()) > 4).astype(int) # this line then checks the amount of indexes next to eachother, if more than 4, it is labelled as clipping.
            
    samplesbefore = 0.1//(1/sampfreq)                                           # 100 ms before the same value needs to be changed
    samplesafter = 0.3//(1/sampfreq)                                            # 300 ms after the same value needs to be changed
    clippingdf = ecg_df[ecg_df['clipping'] == 1]

    mediansignal = ecg_df['ecg_signal'].median()
    indices = clippingdf.index
    for index in indices:               # change all values accordingly to their labels
        ecg_df.loc[index-samplesbefore:index, 'clipping'] = 1
        ecg_df.loc[index-samplesbefore:index,'ecg_signal'] = mediansignal
        ecg_df.loc[index:index+samplesafter, 'clipping'] = 1
        ecg_df.loc[index:index+samplesafter, 'ecg_signal'] = mediansignal
                
    clippingdf2 = ecg_df[ecg_df['clipping'] == 1]
    return ecg_df, clippingdf2

def pacespike(ecg_df):
    '''
    Function to count the amount of values that have a high slope. This is a
    characteristic of a pacing spike and should therefore be registered. 

    Parameters
    ----------
    ecg_df : Dataframe
        Dataframe with all the data of the ECG signal [mV].

    Returns
    -------
    pacing : integer
        Count of how many pacing spikes have been detected.

    '''

    ecg_df['Slope'] = ecg_df.ecg_signal.diff()
    pacing = 0
    for value in ecg_df.Slope:
        if value >= 1 or value <= -1:
            pacing += 1
    return pacing
  
def ecg_rpeak(ecg_df, sampfreq):
    '''
    Function that uses bioSPPY toolbox in order to filter the ECG signal and
    detect R-peaks. NN intervals are calculated as well.
    Reference: https://github.com/PIA-Group/BioSPPy/blob/5fbfb1de5917eef2609ce985307546d37aa0a0ca/biosppy/signals/ecg.py 

    Parameters
    ----------
    ecg_df : dataframe
        Rows = time [s] and columns = ECG data [mV].
    sampfreq : integer
        Sampling frequency of recording [Hz].

    Returns
    -------
    dataframe : dataframe
        Ecg_df dataframe with added column with filtered ECG signal [mV].
    r_peaks : array
        Contains all detected R-peaks [s].
    ecg_filtered : array
        Contains filtered data of the ECG signal [mV].
    clipping : Dataframe
        Contains all the values for the samples that have been removed due to 
        clipping or no signal and their surrounding.
    pacing : Integer
        Counting of how many times pacing occured within a window.
    ecg_analysis : Integer
        Value to see if there were heartbeats detected. Will be 1 if an error
        occured due to removal of all the heartbeats.
    valid_signal : Boolean
        True if ECG data has been succesfully analysed, False if invalid signal.

    '''
    
    ecg_analysis = 0
    ecg_filtered = ()
    r_peaks = ()
    valid_signal = True

    try:
        ecg_filtered, r_peaks = biosppy.signals.ecg.ecg(ecg_df.ecg_signal, sampfreq, 
                                                        show=False)[1:3]            # [1:3] To get filtered signal and R-peaks
        ecg_df = ecg_df.assign(ecg_filtered = ecg_filtered)                         # Add filtered ECG signal to dataframe
        r_peaks =  r_peaks * 1/sampfreq                                             # Convert from index to time in seconds
    except:
        print('No heartbeats')
        ecg_analysis = 1
        valid_signal = False

    return ecg_df, r_peaks, ecg_filtered, ecg_analysis, valid_signal

def peakprominence(r_peaks, dataframe, ecg_filtered, sampling_rate):
    '''
    Function to see if all R-peaks have a prominence above 2x 85th percentile 
    of total prominences. 

    Parameters
    ----------
    r_peaks : array
        Contains all detected R-peaks [s].
    dataframe : dataframe
        ecg_df with all the columns.
    ecg_filtered : TYPE
        DESCRIPTION.
    sampling_rate : integer
        Sampling rate at which the data was sampled.

    Returns
    -------
    r_peak : array
        Contains all detected R-peaks corrected for the peak prominence [s].
    nni : array
        Contains all calculated NN intervals between the R-peaks [ms].
    noisepeak : integer
        Amount of R-peaks removed due to peak prominence.

    '''
    
    peaks_all, _ = scipy.signal.find_peaks(ecg_filtered)
    prominences = scipy.signal.peak_prominences(ecg_filtered, peaks_all)[0]
    
    prominence_85 = np.percentile(prominences, 85)
    r_peaks_prom_85, _ = scipy.signal.find_peaks(ecg_filtered,prominence=(prominence_85))
    r_peaks_prom_85 = r_peaks_prom_85 * 1/sampling_rate
    
    r_peak_verified = []
    for element in r_peaks:
        if element in r_peaks_prom_85:
            r_peak_verified.append(element)
    #print(f'this is the length of verified rpeaks: {len(r_peak_verified)}')
    
    nni = tools.nn_intervals(r_peak_verified)
    #print(f'this is the length of nni peaks: {len(nni)}')

    noisepeak = len(r_peaks) - len(r_peak_verified)

    return r_peak_verified, nni, noisepeak

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
    
    nni = pd.Series(nni)
    r_peaks = pd.Series(r_peaks)
    nni_medians = nni.rolling(10, min_periods=1).median()
    index_ect = []
    
    for index, item in enumerate(nni):
        if (item > 6000) | (item < 300):            #non physiological values
            index_ect = index_ect + [index]
        elif (item < 0.75*nni_medians[index]):      # ectopic beat will be too soon
            try:
                if (nni[index+2] > 0.8*nni_medians[index]): # and the next will be longer
                    index_ect = index_ect + [index]         # save index
            except: 
                index_ect = index_ect + [index]             # if no next value exists, (at the end) this prevents error

    for i in index_ect:
        nni = nni.drop(index=i)         #remove nni's

    for i in index_ect:
        try:
            nni[i] = (nni[i-1] + nni[i+1])/2    # replace with interpolated nni from 1 before and 1 after
            nni = nni.drop(index=i+1)           #remove next nni. since the ectopic beat altered 2 nni's
            r_peaks = r_peaks.drop(index=i+1)   # and remove corresponding rpeaks
        except:
            nni[i] = nni_medians[i]             # to prevent error at the end
            r_peaks = r_peaks.drop(index=i+1)       
    
    medianinterval = nni.median()
    
    for index, value in enumerate(nni):
        if (value < 0.75 * medianinterval) | (value > 1.25 * medianinterval): #check nni's are within range
            try:
                nni = nni.drop(index = index)
                r_peaks = r_peaks.drop(index=index+1)
            except:
                continue

    return r_peaks, nni

def ecg_check(noisepeak, pacing, clipping, ecg_analysis, windowmin, sampfreq, nni):
    """
    Checks whether any statements are met that make the signal invalid. Then change valid_signal to false.
    Makes sure next time window is appropriately started.
    Thresholds are empirically found, are open to changes.

    Parameters
    ----------
    noisepeak:
        Amount of R-peaks removed due to peak prominence.
    pacing:
        amount of pacing instances found (samples)
    clipping:
        amount of clipping found
    ecg_analysis
        is 1 if ECG loading went wrong.
    windowmin:
        amount of minutes per window
    sampfreq:
        sample frequency of MIMIC-IV in Hz
    nni:
        lst containing all nni's

    Output
    ----------
    valid_signal: Boolean 
                  Containing 'True' if the signal is valid and of good quality
                  Next time segment will be started if 'False'.
    """
    bad_signal = 0.25*60*windowmin*sampfreq
    removed = len(clipping) + noisepeak
    valid_signal = True

    if pacing > 100:                
        valid_signal = False
        print(f'try next time segment: pacing detected (#pacing = {pacing})')
    elif removed > bad_signal:
        valid_signal = False
        print(f'try next time segment: bad signal quality')
    elif len(nni) == 0:
        valid_signal = False
        print(f'try next time segment: no NN-intervals detected')
            
    return valid_signal
    
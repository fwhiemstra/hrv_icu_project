#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Artefact) Detection
Created: 10/3/2022 - 05/2022
Last update: 23/02/2023
Python v3.9.5('base': conda)
Adjusted: M. ter Braak
"""
import numpy as np
import pandas as pd
from math import ceil
import biosppy
import scipy
from pyhrv import tools
from scipy.signal import welch
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# %% Functions

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
    Reference: https://github.com/PIA-Group/BioS4PPy/blob/5fbfb1de5917eef2609ce985307546d37aa0a0ca/biosppy/signals/ecg.py 

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
    r_peaks_prom_85, _ = scipy.signal.find_peaks(ecg_filtered, prominence=(prominence_85))
    r_peaks_prom_85 = r_peaks_prom_85 * 1/sampling_rate
    
    r_peak_verified = []
    for element in r_peaks:
        if element in r_peaks_prom_85:
            r_peak_verified.append(element)
    #print(f'this is the length of verified rpeaks: {len(r_peak_verified)}')
    
    # nni = tools.nn_intervals(r_peak_verified)
    #print(f'this is the length of nni peaks: {len(nni)}')

    noisepeak = len(r_peaks) - len(r_peak_verified)
    noisepeak_val = list(set(r_peaks) - set(r_peak_verified))
    return r_peak_verified, noisepeak, noisepeak_val

def movement_art(ecg_df, r_peaks_first, sampfreq):
    '''
    Function to detect baseline shifts, possibly indicating movement artefacts. Movement artefacts have a frequency around 0-1 Hz,
    whereas the QRS is more in the range of 0-40 Hz. 

    Parameters
    ----------
    r_peaks : array
        Contains all detected R-peaks [s].
    ecg_df : dataframe
        ecg_df with all the columns.
    sampfreq:
        in Hz
    
    Returns
    -------
    selection:
        Contains an array with 0 and 1 whether a baseline shift is detected or not.
    nni:
        Calculated nni's after correct detection of the R-peaks  
    time:
        Array with the detected R-peaks
    '''
    bas_shift = []
    count = []
    for second in range(0,len(ecg_df['index']),1500): # 1500 indexes in two intervals (6 sec)
        signal = ecg_df['ecg_signal']
        signal = np.array(signal)
        if second+3000 <= len(ecg_df['index']):
            interval = range(second, second+3000) # periods of 12 sec for analyses
            ecg_tijdsstap = (list(signal[interval]))
        else:
            interval = range(second, len(ecg_df['index'])) # for the last segments
            ecg_tijdsstap = (list(signal[interval]))

        f_12, Pxx_12 = welch(ecg_tijdsstap, fs=sampfreq, nperseg=2048, scaling="spectrum")
        Pxx_rel_12 = Pxx_12/sum(Pxx_12)
        plt.plot(f_12, Pxx_rel_12)
        k = []
        k = [k.append(i) for i in f_12 if i <= 1] # spectrum of 0-1 Hz
        l = []
        l = [l.append(i) for i in f_12 if i <= 40] # spectrum of 0-40 Hz

        # Calculate integral and baseline shift
        integral = cumulative_trapezoid(Pxx_rel_12[0:len(l)], f_12[0:len(l)], initial = 0)
        basSQI_12 = (integral[len(l)-1]-integral[len(k)])/integral[len(l)-1]

        if basSQI_12 <= 0.60: # threshold for baseline shift, can be adapted (0.8 was too high)
            count = [interval[1]]
            new_count = [x/ sampfreq for x in count][0]
            bas_shift.append(new_count)
    print(f'There are {len(bas_shift)} baseline shifts detected')

    # remove nni's which fall within a baseline shift interval
    nni = tools.nn_intervals(r_peaks_first)
    time = r_peaks_first
    time = np.array(r_peaks_first)
    selection = np.ones(len(time))
    for b in bas_shift:
        selection = np.logical_and(selection, np.logical_not(np.logical_and((time >= b), (time <= b+6.1)))) #6.1 seconds, otherwise the point at 6 seconds exactly will be left in
        
    # print(selection) # array with true and false whether that R-peak falls in a baseline shift interval
    if bas_shift:
        nni = nni[selection[:-1]]
    # print(nni) # Selected nni's in the period without baseline shifts
    return bas_shift, selection, nni, time


def ecg_ectopic_removal(r_peaks, nni):
    """
    Function for the removal of outliers and ectopic beats.
    
    INPUT:
        r_peaks: array containing all detected R-peaks [s]
        nni: array containing all calculated NN intervals [ms]
        
    OUTPUT:
        nni_true: corrected array containing all NN intervals [ms] 
        
    """
    
    nni = pd.Series(nni)
    r_peaks = pd.Series(r_peaks)
    r_peaks_first = r_peaks
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
    
    ectopic_beat = list(set(r_peaks_first) - set(r_peaks)) # R-peaks removed based on ectopic beat
    # print(ectopic_beat)
    # print(r_peaks)
    return r_peaks, nni, ectopic_beat

def ecg_check(noisepeak, pacing, clipping, ecg_analysis, windowmin, sampfreq, nni, selection, r_peaks, noisepeak_val, ectopic_beat):
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
    ectopic beat:
        Contains the time of ectopic beats
    noisepeak_val:
        Contain the time of the noise peak
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
    selection:
        containing the baseline shifts

    Output
    ----------
    valid_signal: Boolean 
                  Containing 'True' if the signal is valid and of good quality
                  Next time segment will be started if 'False'.
    """
    bad_signal = 0.25*60*windowmin*sampfreq
    removed = len(clipping)
    valid_signal = True

    if pacing > 100:                
        valid_signal = False
        print(f'try next time segment: pacing detected (#pacing = {pacing})')
    elif removed > bad_signal:
        valid_signal = False
        print(f'try next time segment: bad signal quality')
    elif len(list(filter(None, selection))) < 0.75*len(selection): # check whether <25% is removed
        valid_signal = False
        print(f'try next time segment: signal contains too much baseline shifts')
    elif (len(noisepeak_val)+len(ectopic_beat)) > 0.10*(len(r_peaks)+len(noisepeak_val)+len(ectopic_beat)):  # check whether 10% of the peaks are detected as ectopic or noise
        valid_signal = False
        print(f'try next time segment: too much noise and ectopic beats detected')
    elif len(nni) == 0:
        valid_signal = False
        print(f'try next time segment: no NN-intervals detected')
            
    return valid_signal
    
# %%

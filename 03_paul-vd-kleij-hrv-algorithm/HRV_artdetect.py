#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artefact detection
Created: 103/2022 - 05/2022
Python v3.8
Author: F. Smits
"""
import numpy as np
import pandas as pd
from math import ceil
#%% Functions

def findsame(ecg_df, sampfreq):
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
    labels = (ecg_df.ecg_signal != ecg_df.ecg_signal.shift()).cumsum()
    ecg_df['same_value'] = (labels.map(labels.value_counts()) >= 5).astype(int)
    
    samplesbefore = 0.1//(1/sampfreq)                                           # 100 ms before the same value needs to be changed
    samplesafter = 0.3//(1/sampfreq)                                            # 300 ms after the same value needs to be changed
    clippingdf = ecg_df.loc[ecg_df['same_value'] == 1]

    mediansignal = ecg_df['ecg_signal'].median()
    indices = clippingdf.index
    for index in ecg_df.index:
        if index in indices:
            ecg_df.loc[index-samplesbefore:index, 'same_value'] = 1
            ecg_df.loc[index-samplesbefore:index,'ecg_signal'] = mediansignal
            ecg_df.loc[index:index+samplesafter, 'same_value'] = 1
            ecg_df.loc[index:index+samplesafter, 'ecg_signal'] = mediansignal
                
    clippingdf2 = ecg_df.loc[ecg_df['same_value'] == 1]
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

            
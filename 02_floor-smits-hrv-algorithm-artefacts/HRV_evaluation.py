# -*- coding: utf-8 -*-
"""
HRV analysis
Created: 10/2021 - 02/2022
Python v3.8
Author: M. Verboom
"""

#%% Required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Visual evaluation of R peaks
def visual_evaluation_rpeaks(batch_df, batch_rpeaks):
    """
    Function that creates a plot for the visual evaluation of R-peak detection.
    The plot that is created shows the raw ECG signal with all detected R-peaks
    by the algorithm. 
    
    INPUT:
        batch_df: list containing dataframes with raw ecg,
                          filtered ecg and time of all patients
        batch_rpeaks: list containing all locations of r-peaks for all 
                      patients [s]   
    
    OUTPUT:
        plot with raw ECG signal and detected R-peaks
    """
    
    
    for i in np.arange(0, len(batch_df), 1):
        dataset = batch_df[i]
        data = dataset.ecg_signal
        time = dataset.Time
        time = [x-time.iloc[0] for x in time]
        rpiek = batch_rpeaks[i]
    
        plt.figure()
        plt.plot(time, data, 'c', label='Raw ECG signal')
        plt.axvline(rpiek[1], color='m', label='Detected R-peaks')
        plt.xlabel('Time [s]', fontsize = 15)
        plt.ylabel('Amplitude [mV]', fontsize = 15) 
        plt.legend(loc=2, prop={'size': 20})
        
        # Plot vertical lines at every detected R-peak location
        for i in np.arange(0, len(rpiek), 1):
            plt.axvline(rpiek[i], color = 'm')
        
#%% Visual evalation of NNI series

def visual_evaluation_nni(batch_nni, batch_rpeaks, batch_nni_first,
                          batch_rpeaks_first):
    """
    Function that creates a plot for the visual evaluation of NNI correction.
    The plot that is created shows the NNI series before and after removal of
    ectopic beats and outliers. 
    
    INPUT:
        batch_nni: list containing arrays with nni per included patient [ms]
        batch_nni_first: list containing arrays with nni per included patient 
                         [ms], before ectopic beat- and outlier removal
        batch_rpeaks: list containing arrays with rpeak locations [s]
        batch_rpeaks_first: list containing array with rpeak locations [s]-
                            before ectopic beat- and outlier removal
    
    OUTPUT:
        plot with NNI series before and after ectopic beat- and outlier removal
    """
    
    for i in np.arange(0, len(batch_nni), 1):
        rpeak = batch_rpeaks[i]
        rpeak_first = batch_rpeaks_first[i]
        nni = batch_nni[i]
        nni_first = batch_nni_first[i]
        plt.figure()
        plt.plot(rpeak_first[:-1], nni_first, 'm-o',
                 label='Before ectopic beat removal')
        plt.plot(rpeak, nni, 'c-o', label='After ectopic beat removal')
        plt.xlabel('Time of R-peak [s]', fontsize=20)
        plt.ylabel('NNI [ms]', fontsize=20)
        plt.legend(loc=2, prop={'size': 20})
        
#%% Histogram of all calculated HRV parameters        
        
def hrv_distribution(dfhist):
    """
    Function that creates histograms of all calculated HRV parameters. Mean 
    value and standard deviation are illustrated on the computed histograms.
    
    INPUT: 
        dfhist: string of .csv file name with all HRV parameters
        
    OUTPUT: 
        histogram per HRV parameter
        result: DataFrame containing mean value and standard deviation per 
                HRV parameters   
    """
    
    dfhist = pd.read_csv(dfhist)                                                # Read .csv file
    col = list(dfhist.columns)                                                  # Take names of HRV parameters    
    
    col = col[5:-1]
    col.remove('poincare_plot')
    col.remove('dfa_plot')

    unit = ['time [ms]', 'time [ms]', 'time [ms]', 'x', 'x', 'x',       # Units per HRV parameter, for plotting purposes
            '[bpm]', '[bpm]', '[bpm]', '[bpm]', '[ms]', '[ms]', '[ms]', '[ms]',
            '[ms]','count', '[%]', '[-]', '[ms2]', '[ms2]', '[ms2]', '[ms2]',
            '[ms2]', '[ms2]', '[ms2]', '[ms2]', '[-]', '[-]', '[-]', '[-]',
            '[%]', '[%]', '-', '[ms2]', '[ms]', '[ms]', '[-]', '[-]', '[-]',
            '[beats]', '[beats]' ]
   
    # Create empty lists for mean and standard deviation
    mean = list()                           
    std = list()
    
    for i in np.arange(0, dfhist.shape[1]-8, 1):
        a = dfhist[col[i]].mean()
        b = dfhist[col[i]].std()
        mean.append(a)
        std.append(b)
        # print(mean)
        # print(std)

        
        # Create histogram per column of dfhist
        plt.figure()
        dfhist.iloc[:,i+5].hist(alpha=0.5, color = 'c', bins=20)
        plt.title(col[i], fontsize=20)
        plt.xlabel(unit[i], fontsize=12)
        plt.ylabel('count [-]', fontsize=12)
        plt.axvline(mean[i], color='m', alpha=0.7, label='mean value')
        plt.axvspan((mean[i]-std[i]), (mean[i]+std[i]), alpha=0.2,
                    color='plum', label='standard deviation')
        plt.legend()
        plt.savefig(col[i]+ '.png')
    

    # Create DataFrame containing mean and standard deviation per HRV parameter
    result = pd.DataFrame(data=[mean,std]) 
    result = result.set_axis(col, 'columns') 
    result = result.T 
    result = result.set_axis(['Mean', 'Std'], 'columns')
    
    return result
        

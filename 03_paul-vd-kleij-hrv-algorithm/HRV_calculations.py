# -*- coding: utf-8 -*-
"""
HRV calculations
Created: 10/2021 - 02/2022
last update: 11/11/2022
Python v3.9.5 (base: conda)
Author: M. Verboom
Adjusted: P.A. van der Kleij
"""

# HRV toolbox
from pickle import FALSE
import pyhrv.nonlinear as nl
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
from pyhrv.utils import join_tuples

# biosppy
import biosppy

#%% Created functions

def hrv_calculations(nni, sampfreq, windowmin):
    """
    Function that uses pyHRV toolbox to calculate HRV parameters.
    Reference: https://github.com/PGomes92/pyhrv/tree/master/pyhrv
    
    INPUT:
        nni: array with nni series of a patient [ms]
        sampfreq: sampling frequency of recording [Hz]
    
    OUTPUT:
        results_td: ReturnTuple containing HRV parameters of Time Domain 
        results_fd: ReturnTuple containing HRV parameters of Frequency Domain 
        results_nl: ReturnTuple containing HRV parameters of Nonlinear analysis
        are then combinined into 1 tuple: hrv_parameters
        
    """
    
    # Time domain parameters
    nnpar = td.nni_parameters(nni=nni)                                           # n of intervals, mean, min and max nn interval
    nndif = td.nni_differences_parameters(nni=nni)                               # n of interval differences, mean, min and max of nn interval differences
    hr = td.hr_parameters(nni=nni)                                               # mean, min, max std of HR
    sdnn = td.sdnn(nni=nni)                                                      # standard deviation of NN interval series
    sdnni = td.sdnn_index(nni=nni, full=True, duration=windowmin*60/5, warn=True)          # mean of std of all NN intervals within 5 minute intervals [ms] Needed correction for segmental calculations
    sdann = td.sdann(nni=nni, full=True, overlap=False, duration=windowmin*60/5,           # std of the mean nni value of each segment. Needed correction for segmental calculations    
                     warn=True)
    rmssd = td.rmssd(nni=nni)                                                    # root mean of square differences of successive NN intervals
    sdsd = td.sdsd(nni=nni)                                                      # std of differences of successive NN intervals
    nn50 = td.nn50(nni=nni)                                                      # NN differences > 50 ms                                              
    triang = td.triangular_index(nni=nni, binsize=7.8125, plot=False,            # triangular index bas on NN interval histogram
                                 show=False)                        
    results_td = join_tuples(nnpar, nndif, hr, sdnn, sdnni, sdann,   # Create ReturnTuple containing all time domain parameters
                                         rmssd, sdsd, nn50, triang)
       
         
    # Frequency domain
    fbands={'ulf': (0.00, 0.003), 'vlf': (0.003, 0.04), 'lf': (0.04, 0.15),
            'hf': (0.15, 0.4)}
    results = fd.welch_psd(nni=nni, fbands=fbands, show=False, mode='normal')
   
    # Unwrap ReturnTuple object results, only take desired paramaters 
    abs_ulf = results[2][0]
    abs_vlf = results[2][1]
    abs_lf = results[2][2]
    abs_hf = results[2][3]
    rel_ulf = results[3][0]
    rel_vlf = results[3][1]
    rel_lf = results[3][2]
    rel_hf = results[3][3]
    log_ulf = results[4][0]
    log_vlf = results[4][1]
    log_lf = results[4][2]
    log_hf = results[4][3]
    norm_lf = results[5][0]
    norm_hf = results[5][1]
    ratio_lf_hf = results[6]
    total_power = results[7]
    results_fd = biosppy.utils.ReturnTuple((abs_ulf, abs_vlf, abs_lf, abs_hf,   # Create ReturnTuple containing all frequency domain parameters
                                           rel_ulf, rel_vlf, rel_lf, rel_hf,
                                           log_ulf, log_vlf, log_lf, log_hf,
                                           norm_lf, norm_hf, ratio_lf_hf, 
                                           total_power), ('abs_ulf', 'abs_vlf',
                                                          'abs_lf', 'abs_hf',
                                                          'rel_ulf', 'rel_vlf',
                                                          'rel_lf', 'rel_hf',
                                                          'log_ulf', 'log_vlf',
                                                          'log_lf', 'log_hf',
                                                          'norm_lf', 'norm_hf',
                                                          'ratio_lf_hf',
                                                          'total_power'))
    
    # Nonlinear analysis
    poincare = nl.poincare(nni=nni, show=False)                    # Compute poincare plot and parameters
    entropy = nl.sample_entropy(nni)                               # Calculate sample entropy
    dfa = nl.dfa(nni, show=False)                                  # Perform DFA analysis
    results_nl = join_tuples(poincare, entropy, dfa)

    # Join time domain, frequency domain and non-lineair analysis together
    hrv_parameters = join_tuples(results_td, results_fd, results_nl)

    return hrv_parameters

def to_dataframe(hrv_parameters, ind_time, lst_all):
    """
    Creates a dataframe from hrv parameters. Each time HRV parameters are calculated from a segment,
    they are added to the list, that will later be converted into a dataframe.

    Input:
    -----
    hrv_parameters: 
        df containin all hrv_parameters for that segment
    ind_time:
        Index of segment
    lst_all:
        lst containing all previous hrv_parameters

    Output:
    -------
    lst_all:
        lst containing all hrv_parameters, including current segment
    tuplekeys:
        lst containing tuple keys for all hrv_parameters. 
    """
    if ind_time == 0:
        lst_all = []
    
    tuplekeys = hrv_parameters.keys()
    lst = []

    for key in tuplekeys:
        lst += [hrv_parameters[key]]
    lst_all.append(lst)
    
    return lst_all, tuplekeys
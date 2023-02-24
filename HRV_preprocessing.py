# -*- coding: utf-8 -*-
"""
HRV preprocessing
Created: 10/2021 - 02/2022
Adjustments: 22/11/2022
Python v3.9.5 (base: conda)
Author: M. Verboom
Adjusted by: F. Smits, P.A. van der Kleij
"""
#%%
# Required modules
# Basic import
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Signals toolbox
import biosppy
#import pyhrv.tools as tools

# import the WFDB(WaveFormDataBase) package
import wfdb

import datetime
import scipy
import matplotlib.pyplot as plt

# %%
def create_window(startmin, windowmin, segmentmin, segmentamount):
    '''
    Creates the first window start and stoptimes  within all segments

    Parameters
    ----------
    startmin : int
        After how many minutes into the signal do you want to start the HRV analysis
    windowmin : int
        How many minutes of signal do you want to use in each analytical window
    segmentmin : int
        How many minutes do you want a segment to be. Note that 1 segment translates into 1 datapoint
        the segmentmin gives the interval between datapoints and thus your sample frequency in the
        cosinor analysis
    segmentamount :
        the amount of segments you want to analyse. total analysis time is segmentmin*segmentamount.


    Returns
    -------
    starttime : list
        List with starttimes in seconds of analytical windows.
    endtime : array
        List with endtimes in seconds of analytical windows.

    '''
    segment_list = list(range(segmentamount))
    starttime = []
    stoptime = []

    for segment in segment_list:
        starttime.append(segmentmin*(startmin + segment*segmentmin))
        stoptime.append(segmentmin*(startmin + windowmin + segment*segmentmin))

    return(starttime, stoptime)

def load_pts(pt_records_file):
    """
    Search directory for patient file to load data

    Input:
    -------
    pt_records_file : string
        this is the filename containing rows with all patient ID's
        as an example, refer to pt_records_mimic4.txt

    Output:
    -------
    pt_records: lst
        list with all patient ID's 
    """
    cwd = os.getcwd()
    cwd_path = Path(cwd)
    for item in cwd_path.glob(f'**/{pt_records_file}'):
        with open(item) as f:
            pt_records = [x.strip() for x in list(f) if x]
    return(pt_records)


def load_data(pt_record, sampfreq, start, stop, lead='II'):
    """
    Function to load data from MIMIC-IV database. it streams the data
    from physionet.org.
    
    INPUT:
        pt_record: ID number of patient for analysis
        sampfreq: sampling frequency of recording [Hz]
        start: starting time of ECG analysis
        stop: ending time of ECG analysis
        lead: ECG lead for analysis: 'I', "II", "V"
        
    OUTPUT:
        ecg_data: data from waveform database
        pt_path: patient path in directory
        w_record_name: name of patient directory on computer
        database_name: name of database searched
        record_dir: final patient directory for MIMIC-IV
        w_record_header: header file containing info on the record
    """

    start_frame = int(sampfreq * start)//4                                           # The starting frame number to read for all channels. Notice the //4 to change samples to frames
    stop_frame = int(sampfreq * stop)//4                                             # The sample frame at which to stop reading for all channels. notice the //4 to change samples to frames.
    pt_path = Path(pt_record)
    w_record_name = pt_path.name 
    database_name = 'mimic4wdb/0.1.0'
    record_dir = f"{Path(database_name)}\\{pt_path.parent}".replace('\\', '/')       # final directory that should be searched for in MIMIC-IV

    w_record_header = wfdb.rdheader(w_record_name, pn_dir=record_dir, rd_segments=True)     #keep rd_segments true to assess all segments at once
    ecg_data = wfdb.rdrecord(w_record_name, pn_dir=record_dir, m2s=True,                           #keep m2s true to combine all segments 
             sampfrom=start_frame, sampto=stop_frame, channel_names=lead, smooth_frames=False)     #keep smooth_frame False to look at samples instead of frames.
    
    return ecg_data, pt_path, w_record_name, database_name, record_dir, w_record_header


def ecg_dataframe(ecg_record, sampfreq):
    """
    Function to create dataframe with timestamps and raw ECG data

    INPUT:
        ecg_record: data from waveform database
        sampfreq: sampling frequency of recording [Hz]
        
    OUTPUT:
        ecg_df: DataFrame with rows = time [s], columns = ECG data [mV]
    """   
    t = np.linspace(0, ecg_record.sig_len*4, ecg_record.sig_len*4)                          # Length of signal
    t_and_p_sig = np.column_stack((t,ecg_record.e_p_signal[0]))                             # extract signal itself and time axis
    ecg_df = pd.DataFrame(t_and_p_sig, columns=('Time','ecg_signal'))                                                      # Create dataframe for plotting
    
    # NaN removal
    index = np.arange(0, ecg_df.ecg_signal.first_valid_index(), 1)
    ecg_df=ecg_df.drop(labels=index, axis=0)        
    ecg_df = ecg_df.fillna(0)
    ecg_df.Time = ecg_df.Time/sampfreq                                          # Time axis in seconds (~250 Hz)
    

    timestart = ecg_record.base_time
    d = timestart.strftime("%d")
    h = timestart.strftime("%H")
    m = timestart.strftime("%M")
    s = timestart.strftime("%S")

    ticu = datetime.timedelta(days = int(d), hours = int(h), minutes = int(m), seconds = int(s))  #Days are not always correctly notated in MIMIC-IV. See cosinor analysis for correction
    df_timemoment = pd.DataFrame(columns=(['TimeStamp']))

    for i in ecg_df['Time']:
        timemoment = ticu + pd.Timedelta(seconds = int(i))
        df_timeadd = pd.DataFrame({'TimeStamp': [timemoment]})
        df_timemoment = df_timemoment.append(pd.DataFrame(df_timeadd), ignore_index = True)
    ecg_df = ecg_df.reset_index()
    ecg_df['TimeStamp'] = df_timemoment['TimeStamp']
    return ecg_df

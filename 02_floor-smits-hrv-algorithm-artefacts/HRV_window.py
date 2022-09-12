#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRV analysis
Created: 03/2022 - 05/2022
Python v3.8.8
Author: F. Smits
"""

#%% Required modules
import numpy as np


#%% 
def create_window(HRVwindow, Start, Duration):
    '''

    Parameters
    ----------
    HRVwindow : int
        Specified window of analysis of the HRV in minutes. 
        This window is the window in which the different parameters get 
        determined.
    Start : int
        After how many minutes of the start of the signal the analysis musts 
        start.
    Duration : int
        For how long of the signal the analysis must endure.

    Returns
    -------
    starttime : array
        Array used in the further analysis to define the different start time,  
        duration and the window.
    endtime : array
        Array used in the further analysis to define the different end times 
        and the window.

    '''
    steps = 60*HRVwindow
    startsec = 60*Start
    dursec = 60*Duration
    dursec += startsec
    
    controlfloor = Duration//HRVwindow
    controldec = Duration/HRVwindow
    if controlfloor == controldec:
        dursec = dursec
    else:
        dursec = controlfloor*steps + startsec
    
    endsec = startsec+steps
    enddur = dursec+steps
    
    starttime = np.arange(start = startsec, stop = dursec, step = steps)
    endtime = np.arange(start = endsec, stop = enddur, step = steps)
    return (starttime, endtime)

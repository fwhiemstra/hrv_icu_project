#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R peak validation
Created: 10-01-2023
Last update: 23-02-2023
Python v3.9.5('base': conda)
Author: M. ter Braak
"""

# Required modules
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# %% Function

def visualisation(ecg_df, r_peaks, time, selection, ectopic_beat, noisepeak_val, windowmin, segmentmin):

    '''
    Function to show the located R-peaks with the ECG signal for visual validation.

    INPUT
    ecg_df: to get the original ECG signal
    r_peaks: to visualise the detected R-peaks with the exclusion of the artefacts
    time: detected R peaks
    selection: list with presenence of baseline shifts
    ectopic beat: time of the ectopic beats
    noisepeak_val: time of the noise peaks
    windowmin: the length of the window
    segmentmin: the length of the segment analysed

    '''

    t = ecg_df['Time']; t = t.to_numpy()
    s = ecg_df['ecg_signal']; s = s.to_numpy()
    r = r_peaks.to_numpy()
    # y values interpolated based on the ECG signal from the MIMIC-IV database
    y_rpeaks = np.interp(r, t, s)
    y_rpeaks_bas = np.interp(time, t, s)
    y_rpeaks_ect = np.interp(ectopic_beat, t, s)
    y_rpeaks_peak = np.interp(noisepeak_val, t, s)
    
    selection = np.logical_not(selection)
    # Create figure
    fig = go.Figure()
    # All traces added to the visualisation
    fig.add_trace(go.Scatter(x=list(ecg_df.Time), y=list(ecg_df.ecg_signal), name='ECG signal'))
    fig.add_trace(go.Scatter(x=r, y=y_rpeaks, mode='markers', name='Detected R-peak'))
    fig.add_trace(go.Scatter(x=time[selection], y=y_rpeaks_bas[selection], mode='markers', marker=dict(symbol="diamond"), name='Baseline shift R-peak'))
    fig.add_trace(go.Scatter(x=ectopic_beat, y=y_rpeaks_ect, mode='markers', marker=dict(symbol="star-square"), name='Ectopic beat'))
    fig.add_trace(go.Scatter(x=noisepeak_val, y=y_rpeaks_peak, mode='markers', marker=dict(symbol="star-triangle-up"), name='Noise peak'))

    print(f'amount of detected r-peaks = {len(r)}')
    print(f'amount of ectopic r-peaks = {len(ectopic_beat)}')
    print(f'amount of baseline shift r-peaks = {len(time[selection])}')
    print(f'amount of  noise peaks = {len(noisepeak_val)}')

    fig.update_layout(
        title_text="Range slider of ECG signal"
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        )
    )

    fig.update_yaxes(
        title_text='ECG (mV)',
        range=[-1.5, 1.5],
    )
    fig.update_xaxes(
        title_text='Time (s)',
        range=[0, windowmin*segmentmin])

    fig.show()

# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRV analysis
Created: 01/11/2022
Python v3.9.5 ('base': conda)
Author: P.A. van der Kleij, F.W. Hiemstra
"""

# %% Required modules
import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats
import os
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import math

# Heart Rate Variability Evaluation
'''
We'll first load the data, whereafter the time axis is constructed. And unnecessary parameters are dropped.
'''

numberofpatients = 2    # amount of .csv-files you have

for ind_pt in range(1, numberofpatients+1):      # make sure the range alignes with the csv-files
    filename = f'patient_hrvdata_{ind_pt}.csv'
    cwd = os.getcwd()
    cwd_path = Path(cwd)
    for item in cwd_path.glob(f'**/{filename}'):    # search directory for files
        hrv_df = pd.read_csv(item)
    # print (hrv_df)
    time_axis_cosinor = []

    for string in hrv_df['TimeStamp']:
        t = datetime.datetime.strptime(string, "%w days %H:%M:%S")       # in csv file timeconstruct was saved as string. Give time template here
        timedelta = pd.Timedelta(days=t.day, hours=t.hour, minutes=t.minute, seconds=t.second)    # extract time from time template and covert to timedelta element
        time_axis_cosinor += [timedelta/pd.Timedelta(hours=1)]  # convert to hours

    starttime = time_axis_cosinor[0] % 24   # get the starttime in hours of day. i.e. 9 am will be 9, 6pm will be 18

    time_axis_cosinor = [x - time_axis_cosinor[0] for x in time_axis_cosinor]  # remove start from all values
    for j in range(1, len(time_axis_cosinor)):                       # since days sometimes dont correspond, this loop check if next value is bigger than the last
        if time_axis_cosinor[j] <= time_axis_cosinor[j-1]:          # if not, a day should be added to correct it.
            time_axis_cosinor[j] = time_axis_cosinor[j] + 24

    hrv_df = hrv_df.drop(columns=['Unnamed: 0','poincare_plot', 'dfa_plot', 'dfa_alpha1_beats', 'dfa_alpha2_beats','TimeStamp'])

    # %% Cosinor Analysis
    '''
    Cosinor analysis first describes a Sine wave. this sine wave is fitted to the data. then the similarity between datapoints
    and fitted curve is assessed. This analysis happens for every HRV parameter in each patient.
    '''
    def nonlinear_cosinor_function(x, a, b, c):  # definiton for sine wave to fit
        return a + b * np.cos([(2 * np.pi / 24) * (x[j]-c) for j in range(0, len(x))])

    # Set constraints/bounds for parameters
    param_bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])  # the parameters you want to vary. Consider adding boundaries if wished for.

    for ind_key, key in enumerate(hrv_df.keys()):
        # Curve fit
        popt, pcov = optimize.curve_fit(nonlinear_cosinor_function, time_axis_cosinor, hrv_df[key], bounds=param_bounds) # optimize curvefit
        a, b, c = popt      # a, b, c are the parameters in the fitted curve
        a_se, b_se, c_se = np.sqrt(np.diag(pcov))
        y_hat = nonlinear_cosinor_function(time_axis_cosinor, a, b, c)  # this is the fitted curve

        # Circadian parameters
        mesor = a               # a corresponded with the mesor
        mesor_se = a_se

        amplitude = b           # b with the amplitude
        amplitude_se = b_se

        acrophase = c           # c with the (acro)phase
        acrophase_se = c_se

        if amplitude < 0:       # correction for negative sine
            acrophase = acrophase + 12
            amplitude = abs(amplitude)

        if acrophase > 24 or acrophase < 0:  # normalize acrophase between 0 and 24
            acrophase = acrophase % 24

        # Model statistics
        RSS = math.sqrt(np.sum([(hrv_df.at[i, key] - y_hat) ** 2 for i in range(0, len(hrv_df))]))    # root sum square
        MSS = np.sum((y_hat - np.mean([hrv_df.at[i, key] for i in range(0, len(hrv_df))])) ** 2)      # mean sum square
        k = 3  # number of model parameters
        N = len(hrv_df[key])  # degrees of freedom
        F = (MSS / (k - 1)) / (RSS / (N - k))  # f-statistic
        p = 1 - stats.f.cdf(F, k - 1, N - k)  # p value corresponding to f-statistic

        if ind_key == 0:
            cosinor_parameters = pd.DataFrame(columns=['Mesor', 'Mesor SE', 'Amplitude',
                                                    'Amplitude SE', 'Acrophase',
                                                    'Acrophase SE', 'RSS', 'MSS',
                                                    'F statistic', 'p-value'])
        
        newrow = pd.DataFrame({'Mesor': mesor,
                            'Mesor SE': mesor_se,
                            'Amplitude': amplitude,
                            'Amplitude SE': amplitude_se,
                            'Acrophase': acrophase,
                            'Acrophase SE': acrophase_se,
                            'RSS': RSS,
                            'MSS': MSS,
                            'F statistic': F,
                            'p-value': p}, index=[key])
        cosinor_parameters = cosinor_parameters.append(newrow, ignore_index=True)  # load all cosinor parameters for this hrv parameter
        if ind_key == 1:    # by chaning the index here, choose which figure to plot. Python errors with more than 20 plots
            plt.figure()
            plt.plot([t+starttime for t in time_axis_cosinor], hrv_df[key], 'ko', 
                     [t+starttime for t in time_axis_cosinor], y_hat, 'r')
            plt.title(f'24-h Cosinor fit of NNI mean, patient {ind_pt}')
            plt.xlabel('Zeitgeber Time (hours)')
            plt.ylabel('NNI mean (ms)')
            plt.savefig(f'NNI_mean_{ind_pt}.png')
            plt.show()

    # cosinor_parameters['Keys'] = [key for key in hrv_df.keys()]   # something to try for export of cosinor parameters. think about which dataformat
    # cosinor_parameters = cosinor_parameters.set_index('Keys')
    # cosinor_parameters.to_csv(f'Cosinor_hrv_pt{ind_pt}.csv')
    # print(cosinor_parameters)

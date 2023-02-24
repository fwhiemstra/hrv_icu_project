#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fragment evaluation 
Last update: 23/02/2022
Python v3.9.5 ('base': conda)
Adjusted: M. ter Braak

Via this function the fragments are analysed. Comparisons could be made with length variations and the implementation of baseline shift.

""" 

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# %% Length fragment 
# 5 min, 300 sec
entire_frag_1 = pd.read_csv('patient_hrvdata_1_5min.csv')
entire_frag_2 = pd.read_csv('patient_hrvdata_2_5min.csv')
entire_frag_3 = pd.read_csv('patient_hrvdata_3_5min.csv')
entire_frag_4 = pd.read_csv('patient_hrvdata_4_5min.csv')
entire_frag_5 = pd.read_csv('patient_hrvdata_5_5min.csv')
entire_frag_6 = pd.read_csv('patient_hrvdata_6_5min.csv')
entire_frag_7 = pd.read_csv('patient_hrvdata_7_5min.csv')
entire_frag_8 = pd.read_csv('patient_hrvdata_8_5min.csv')
entire_frag_9 = pd.read_csv('patient_hrvdata_9_5min.csv')
entire_frag_10 = pd.read_csv('patient_hrvdata_10_5min.csv')
entire_frag_11 = pd.read_csv('patient_hrvdata_11_5min.csv')

entire_frag = pd.concat([entire_frag_2, entire_frag_3,entire_frag_4, entire_frag_6, entire_frag_7, entire_frag_8, entire_frag_9])

# 225 sec
shorter_frag_1 = pd.read_csv('patient_hrvdata_1_225sec.csv')
shorter_frag_2 = pd.read_csv('patient_hrvdata_2_225sec.csv')
shorter_frag_3 = pd.read_csv('patient_hrvdata_3_225sec.csv')
shorter_frag_4 = pd.read_csv('patient_hrvdata_4_225sec.csv')
shorter_frag_5 = pd.read_csv('patient_hrvdata_5_225sec.csv')
shorter_frag_6 = pd.read_csv('patient_hrvdata_6_225sec.csv')
shorter_frag_7 = pd.read_csv('patient_hrvdata_7_225sec.csv')
shorter_frag_8 = pd.read_csv('patient_hrvdata_8_225sec.csv')
shorter_frag_9 = pd.read_csv('patient_hrvdata_9_225sec.csv')
shorter_frag_10 = pd.read_csv('patient_hrvdata_10_225sec.csv')
shorter_frag_11 = pd.read_csv('patient_hrvdata_11_225sec.csv')

shorter_frag = pd.concat([shorter_frag_2,shorter_frag_3,shorter_frag_4, shorter_frag_6, shorter_frag_7, shorter_frag_8, shorter_frag_9])

# %% Baseline shift implementation
# 5 min, without implementation of baseline shift
without_bas_1 = pd.read_csv('patient_hrvdata_1_zbas.csv')
without_bas_2 = pd.read_csv('patient_hrvdata_2_zbas.csv')
without_bas_3 = pd.read_csv('patient_hrvdata_3_zbas.csv')
without_bas_4 = pd.read_csv('patient_hrvdata_4_zbas.csv')
without_bas_5 = pd.read_csv('patient_hrvdata_5_zbas.csv')
without_bas_6 = pd.read_csv('patient_hrvdata_6_zbas.csv')
without_bas_7 = pd.read_csv('patient_hrvdata_7_zbas.csv')
without_bas_8 = pd.read_csv('patient_hrvdata_8_zbas.csv')
without_bas_9 = pd.read_csv('patient_hrvdata_9_zbas.csv')
without_bas_10 = pd.read_csv('patient_hrvdata_10_zbas.csv')
without_bas_11 = pd.read_csv('patient_hrvdata_11_zbas.csv')
without_bas_12 = pd.read_csv('patient_hrvdata_12_zbas.csv')
without_bas_13 = pd.read_csv('patient_hrvdata_13_zbas.csv')
without_bas_14 = pd.read_csv('patient_hrvdata_14_zbas.csv')
without_bas_15 = pd.read_csv('patient_hrvdata_15_zbas.csv')

without_bas = pd.concat([without_bas_1, without_bas_2, without_bas_3, without_bas_4, without_bas_5, without_bas_6, 
    without_bas_7, without_bas_8, without_bas_9, without_bas_10, without_bas_11, without_bas_12, without_bas_13, without_bas_14, without_bas_15])

# 5 min, with implementation of baseline shift
bas_shift_1 = pd.read_csv('patient_hrvdata_1_bas.csv')
bas_shift_2 = pd.read_csv('patient_hrvdata_2_bas.csv')
bas_shift_3 = pd.read_csv('patient_hrvdata_3_bas.csv')
bas_shift_4 = pd.read_csv('patient_hrvdata_4_bas.csv')
bas_shift_5 = pd.read_csv('patient_hrvdata_5_bas.csv')
bas_shift_6 = pd.read_csv('patient_hrvdata_6_bas.csv')
bas_shift_7 = pd.read_csv('patient_hrvdata_7_bas.csv')
bas_shift_8 = pd.read_csv('patient_hrvdata_8_bas.csv')
bas_shift_9 = pd.read_csv('patient_hrvdata_9_bas.csv')
bas_shift_10 = pd.read_csv('patient_hrvdata_10_bas.csv')
bas_shift_11 = pd.read_csv('patient_hrvdata_11_bas.csv')
bas_shift_12 = pd.read_csv('patient_hrvdata_12_bas.csv')
bas_shift_13 = pd.read_csv('patient_hrvdata_13_bas.csv')
bas_shift_14 = pd.read_csv('patient_hrvdata_14_bas.csv')
bas_shift_15 = pd.read_csv('patient_hrvdata_15_bas.csv')

bas_shift = pd.concat([bas_shift_1, bas_shift_2, bas_shift_3, bas_shift_4, bas_shift_5, bas_shift_6,
    bas_shift_7, bas_shift_8, bas_shift_9, bas_shift_10, bas_shift_11, bas_shift_12, bas_shift_13, bas_shift_14, bas_shift_15])

# %%
descriptives_long = []
descriptives_short = []
tt_test = []
without = []
with_bas = []


for (columnName, columnData) in bas_shift.iteritems():
    descript_long_column = entire_frag[columnName].describe()
    descriptives_long.append(descript_long_column)
    descript_short_column = shorter_frag[columnName].describe()
    descriptives_short.append(descript_short_column)
    without_column = without_bas[columnName].describe()
    without.append(without_column)
    with_bas_column = bas_shift[columnName].describe()
    with_bas.append(with_bas_column)
    if isinstance(columnData, str):
        tt_test_column = stats.ttest_rel(bas_shift[columnName], without_bas[columnName])
        tt_test.append(tt_test_column)


descriptives_long = pd.DataFrame(descriptives_long)
descriptives_long.to_csv(f'descriptives_long.csv')
descriptives_short = pd.DataFrame(descriptives_short)
descriptives_short.to_csv(f'descriptives_short.csv')
without = pd.DataFrame(without)
without.to_csv(f'descriptives_withoutbasshift.csv')
with_bas = pd.DataFrame(with_bas)
with_bas.to_csv(f'descriptive_withbaselineshift.csv')
# tt_test = pd.DataFrame(tt_test)
# tt_test.to_csv(f'Statistics_entire_short.csv')
# %%
# nni_mean_stat = stats.ttest_rel(entire_frag['nni_mean'], shorter_frag['nni_mean'])
# hr_mean_stat = stats.ttest_rel(entire_frag['hr_mean'], shorter_frag['hr_mean'])
# print(nni_mean_stat)
# print(hr_mean_stat)
# %% Evaluation length fragment

# print(entire_frag['nni_mean'])
# print(shorter_frag['nni_mean'])
# print(entire_frag['hr_mean'])
# print(shorter_frag['hr_mean'])
# figures comparing the different lengths
plt.suptitle('HRV measurements of 225 vs 300 seconds')
plt.subplot(131)
plt.plot(entire_frag['nni_mean'], shorter_frag['nni_mean'], 'bo')
plt.plot([350, 850], [350, 850], color='0.8')
plt.title('NNI mean')
plt.axis('equal')
plt.ylabel('NNI mean [ms] using 75% fragment length')
plt.xlabel('NNI mean [ms] using 100% fragment length')
plt.subplot(132)
plt.plot(entire_frag['hr_mean'], shorter_frag['hr_mean'], 'bo')
plt.plot([50, 165], [50,165],color='0.8')
plt.title('HR mean')
plt.axis('equal')
plt.ylabel('HR mean [BPM] using 75% fragment length')
plt.xlabel('HR mean [BPM] using 100% fragment length')
plt.subplot(133)
plt.plot(entire_frag['sdnn'], shorter_frag['sdnn'], 'bo')
plt.plot([0,60],[0,60], color='0.8')
plt.title('SDNN')
plt.axis('equal')
plt.ylabel('SDNN using 75% fragment length')
plt.xlabel('SDNN using 100% fragment length')
plt.show()

length_corr = entire_frag['nni_mean'].corr(shorter_frag['nni_mean'])
print(length_corr)
length_corr_hr = entire_frag['hr_mean'].corr(shorter_frag['hr_mean'])
print(length_corr_hr)
length_corr_sdnn = entire_frag['sdann'].corr(shorter_frag['sdann'])
print(length_corr_sdnn)

# # %% Baseline shift
# plt.suptitle('HRV measurements with and without baseline correction')
# plt.subplot(131)
# plt.plot(bas_shift['nni_mean'], without_bas['nni_mean'], 'bo')
# plt.title('NNI mean')
# plt.ylabel('NNI mean [ms] without baseline shift correction')
# plt.xlabel('NNI mean [ms] with baseline shift correction')
# plt.subplot(132)
# plt.plot(bas_shift['nni_mean'], without_bas['nni_mean'], 'bo')
# plt.title('HR mean')
# plt.ylabel('HR mean [BPM] without baseline shift correction')
# plt.xlabel('HR mean [BPM] with baseline shift correction')
# plt.subplot(133)
# plt.plot(bas_shift['sdnn'], without_bas['sdnn'], 'bo')
# plt.title('SDNN')
# plt.ylabel('SDNN without baseline shift correction')
# plt.xlabel('SDNN with baseline shift correction')
# plt.show()

# bas_corr_nni = bas_shift['nni_mean'].corr(without_bas['nni_mean'])
# print(bas_corr_nni)
# bas_corr_hr = bas_shift['hr_mean'].corr(without_bas['hr_mean'])
# print(bas_corr_hr)


# nni_mean_stat = stats.ttest_rel(bas_shift['nni_mean'], without_bas['nni_mean'])
# hr_mean_stat = stats.ttest_rel(bas_shift['hr_mean'], without_bas['hr_mean'])
# sdnn_mean_stat = stats.ttest_rel(bas_shift['sdnn'], without_bas['sdnn'])
# print(nni_mean_stat)
# print(hr_mean_stat)
# print(sdnn_mean_stat)
# %%

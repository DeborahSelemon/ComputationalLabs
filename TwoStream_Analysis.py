#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:04:11 2023

@author: dos515
"""

import csv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from scipy.optimize import curve_fit

#%% - LOADING CSV data - v Scan

v_label =  [2,4,6,8,10,12]

x_time = [[] for _ in range(len(v_label))]
y_fh = [[] for _ in range(len(v_label))]

plt.figure(figsize =(12,7),dpi=1000)
for i in range(len(v_label)):
    with open('/home/dos515/Documents/Comp_Labs/Run2/data_v%s.csv'%(v_label[i]), 'r') as csvfile:
        raw_data = csv.reader(csvfile, delimiter=',')
        data = []
        for row in raw_data:
            data.append(row)
        #print(data)
    
    x_time[i] = data[0]
    y_fh[i] = data[1]
    
    for j in range(len(x_time[i])):
        x_time[i][j] = float(x_time[i][j])
        y_fh[i][j] = float(y_fh[i][j])

#     #plot data
#     plt.plot(x_time[i],y_fh[i], label = 'velocity = %s'%(v_label[i]),linewidth = 2)
#     #plt.yscale('log')
#     plt.xlabel("Time [Normalised]",fontsize =20)
#     plt.ylabel("First harmonic amplitude [Normalised]", fontsize =20)
#     plt.legend(fontsize =20)
#     plt.yticks(fontsize =20)
#     plt.xticks(fontsize=20)
    
# #plt.savefig('/home/dos515/Documents/Comp_Labs/Run2/v_pic_linear.jpg')
# plt.show()

#%% - EXPONENTIAL FIT

peak_heights = [[] for _ in range(len(y_fh))]
peak_positions = [[] for _ in range(len(y_fh))]
peak_spacing = [[] for _ in range(len(y_fh))]
omega = [[] for _ in range(len(y_fh))]
growth_rate = [[] for _ in range(len(y_fh))]
growth_rate_err = [[] for _ in range(len(y_fh))]

guess_length = [[1,0.008,0.5],[1,0.8,1],[1,1.5,1],[1,3,1],[1,3,1],[1,3,1]]

def exponential(x, a, k, b):
    return a*np.exp(x*k) + b

for i in range(len(y_fh)):
    peaks = find_peaks(y_fh[i],height = 1e-5)
    peak_heights[i] = np.array(peaks[1]['peak_heights']) #list of the heights of the peaks
    peak_pos_list = []
    for j in range(len(peaks[0])):
        peak_pos_list.append(x_time[i][peaks[0][j]]) #list of the peaks positions
    peak_positions[i] = np.array(peak_pos_list)
    peak_spacing[i] = np.diff(peak_positions[i])
    
    #ANGULAR FREQUENCY
    omega[i] = (2*np.pi)/(2*np.mean(peak_spacing[i][0:4]))
    
    #FITTING (AND DAMPING RATE)
    popt_exp, pcov_exp = curve_fit(exponential, peak_positions[i], peak_heights[i], p0=guess_length[i])
    #popt_exp[0,1,2] -> pre-exponential factor, decay/damping rate and baseline

    exp_fit = exponential(peak_positions[i], *popt_exp)

    growth_rate[i] = popt_exp[1]
    growth_rate_err[i] = np.sqrt(pcov_exp[1][1])
    
    
    print('Fit %s DONE'%(i))
    
    # plt.figure(figsize =(12,7),dpi=1000)
    # plt.plot(x_time[i], y_fh[i],'k',linewidth =3)
    # plt.plot(peak_positions[i], peak_heights[i], 'rx', markersize = 20)
    # plt.plot(peak_positions[i], exp_fit, 'r',linewidth =3)
    # plt.xlabel("Time [Normalised]",fontsize =20)
    # plt.ylabel("First harmonic amplitude [Normalised]", fontsize =20)
    # plt.yticks(fontsize =20)
    # plt.xticks(fontsize=20)
    # plt.show()
    
#%% - PRINT RESULTS

print('GROWTH RATE')
print(growth_rate)
print('------------')
print('GRWOTH RATE ERROR')
print(growth_rate_err)
    
#%% - COMPARING GROWTH RATE
variables = [growth_rate] 
fig_label = ['Growth_Rate'] 
x = v_label

for i in range(len(variables)):
    plt.figure(figsize =(12,7),dpi=1000)
    plt.plot(x, variables[i], 'rx', markersize = 15)
    plt.xlabel("Velocity [*thermal speed]",fontsize =20)
    plt.ylabel("%s"%(fig_label[i]), fontsize =20)
    plt.yticks(fontsize =20)
    plt.xticks(fontsize=20)
    plt.savefig('/home/dos515/Documents/Comp_Labs/Run2/vel_%s.jpg'%(fig_label[i]))
    plt.show()

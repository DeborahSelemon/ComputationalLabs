#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:22:38 2023

@author: dos515
"""
import csv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from scipy.optimize import curve_fit



#%% - LOADING CSV data - ncell Scan

ncell_label = [10,20,50,100,1000]

x_time = [[] for _ in range(len(ncell_label))]
y_fh = [[] for _ in range(len(ncell_label))]
run_time = [[] for _ in range(len(ncell_label))]

plt.figure(figsize =(12,7),dpi=1000)
for i in range(len(ncell_label)):
    with open('/home/dos515/Documents/Comp_Labs/Run/data_np200000_nc%s.csv'%(ncell_label[i]), 'r') as csvfile:
        raw_data = csv.reader(csvfile, delimiter=',')
        data = []
        for row in raw_data:
            data.append(row)
        #print(data)
    
    x_time[i] = data[0]
    y_fh[i] = data[1]
    run_time[i] = data[2]
    
    for j in range(len(x_time[i])):
        x_time[i][j] = float(x_time[i][j])
        y_fh[i][j] = float(y_fh[i][j])
        run_time[i][j] = float(run_time[i][j])

#     #plot data
#     plt.plot(x_time[i],y_fh[i], label = 'ncell = %s'%(ncell_label[i]),linewidth = 2)
#     plt.yscale('log')
#     plt.xlabel("Time [Normalised]",fontsize =20)
#     plt.ylabel("First harmonic amplitude [Normalised]", fontsize =20)
#     plt.legend(fontsize =20)
#     plt.yticks(fontsize =20)
#     plt.xticks(fontsize=20)
    
# #plt.savefig('/home/dos515/Documents/Comp_Labs/Run/ncell_pic.jpg')
# plt.show()

#%% - LOADING CSV data - nparticle Scan

nparticle_label = [1000,10000,100000,200000,500000,800000,1000000,10000000]

x_time = [[] for _ in range(len(nparticle_label))]
y_fh = [[] for _ in range(len(nparticle_label))]
run_time = [[] for _ in range(len(nparticle_label))]

plt.figure(figsize =(12,7),dpi=1000)
for i in range(len(nparticle_label)):
    with open('/home/dos515/Documents/Comp_Labs/Run/data_np%s_nc20.csv'%(nparticle_label[i]), 'r') as csvfile:
        raw_data = csv.reader(csvfile, delimiter=',')
        data = []
        for row in raw_data:
            data.append(row)
        #print(data)
    
    x_time[i] = data[0]
    y_fh[i] = data[1]
    run_time[i] = data[2]
    
    for j in range(len(x_time[i])):
        x_time[i][j] = float(x_time[i][j])
        y_fh[i][j] = float(y_fh[i][j])
        run_time[i][j] = float(run_time[i][j])

#     #plot data
#     plt.plot(x_time[i],y_fh[i], label = 'npart = %s'%(nparticle_label[i]),linewidth=2)
#     plt.yscale('log')
#     plt.xlabel("Time [Normalised]",fontsize =20)
#     plt.ylabel("First harmonic amplitude [Normalised]", fontsize =20)
#     plt.legend(fontsize =20)
#     plt.yticks(fontsize =20)
#     plt.xticks(fontsize=20)
    
# plt.savefig('/home/dos515/Documents/Comp_Labs/Run/npart_pic.jpg')
# plt.show()

#%% - LOADING CSV data - Length Scan
     
import csv
import matplotlib.pyplot as plt

length_label = [2,4,8,12,16]

x_time = [[] for _ in range(len(length_label))]
y_fh = [[] for _ in range(len(length_label))]
run_time = [[] for _ in range(len(length_label))]

plt.figure(figsize =(12,7),dpi=1000)
for i in range(len(length_label)):
    with open('/home/dos515/Documents/Comp_Labs/Run/data_L%s.csv'%(length_label[i]), 'r') as csvfile:
        raw_data = csv.reader(csvfile, delimiter=',')
        data = []
        for row in raw_data:
            data.append(row)
        #print(data)
    
    x_time[i] = data[0]
    y_fh[i] = data[1]
    run_time[i] = data[2]
    
    for j in range(len(x_time[i])):
        x_time[i][j] = float(x_time[i][j])
        y_fh[i][j] = float(y_fh[i][j])
        run_time[i][j] = float(run_time[i][j])

#     #plot data
#     plt.plot(x_time[i],y_fh[i], label = 'box length = %s pi'%(length_label[i]),linewidth=2)
#     #plt.yscale('log')
#     plt.xlabel("Time [Normalised]",fontsize =20)
#     plt.ylabel("First harmonic amplitude [Normalised]", fontsize =20)
#     plt.legend(fontsize =20)
#     plt.yticks(fontsize =20)
#     plt.xticks(fontsize=20)
    
# plt.savefig('/home/dos515/Documents/Comp_Labs/Run/length_pic_linear.jpg')
# plt.show()
#%% - FINDING PEAKS & plotting

peak_heights = [[] for _ in range(len(y_fh))]
peak_positions = [[] for _ in range(len(y_fh))]
peak_spacing = [[] for _ in range(len(y_fh))]
omega = [[] for _ in range(len(y_fh))]
damp_rate = [[] for _ in range(len(y_fh))]
damp_rate_err = [[] for _ in range(len(y_fh))]
noise_level = [[] for _ in range(len(y_fh))]

guess_length = [[1,0.5,1],[1,0.8,1],[1,1.5,1],[1,3,1],[1,3,1]]

def exponential(x, a, k, b):
    return a*np.exp(-x*k) + b

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
    popt_exp, pcov_exp = curve_fit(exponential, peak_positions[i][0:4], peak_heights[i][0:4], p0=guess_length[i])
    #popt_exp[0,1,2] -> pre-exponential factor, decay/damping rate and baseline

    exp_fit = exponential(peak_positions[i], *popt_exp)

    damp_rate[i] = popt_exp[1]
    damp_rate_err[i] = np.sqrt(pcov_exp[1][1])
    
    noise_level[i] = popt_exp[2]*-1
    
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
print('------------')
print('OMEGA')
print(omega)
print('------------')
print('DAMPING RATE')
print(damp_rate)
print('------------')
print('DAMPING RATE ERROR')
print(damp_rate_err)
print('------------')
print('NOISE LEVEL')
print(noise_level)
print('------------')
print('RUN TIME')
for i in range(len(run_time)):
    print(run_time[i][0])

#%% -  COMPARING RUNS

variables = [omega, damp_rate] #[omega, damp_rate, run_time, noise_level]
fig_label = ['Angular_Frequency', 'Damping_Rate'] #['Angular_Frequency', 'Damping_Rate', 'Run_Time','Noise_Level']
x = length_label #ncell_label, nparticle_label, length_label

for i in range(len(variables)):
    plt.figure(figsize =(12,7),dpi=1000)
    plt.plot(x, variables[i], 'rx', markersize = 15)
    plt.xlabel("Length of box [*pi]",fontsize =20)
    plt.ylabel("%s"%(fig_label[i]), fontsize =20)
    plt.yticks(fontsize =20)
    plt.xticks(fontsize=20)
    plt.savefig('/home/dos515/Documents/Comp_Labs/Run/length_%s.jpg'%(fig_label[i]))
    plt.show()

#%% - PLOT PEAK FINDING

plt.figure(figsize =(12,7),dpi=1000)
plt.plot(x_time[0], y_fh[0],linewidth =3)
plt.plot(peak_positions[0], peak_heights[0], 'rx', markersize = 20)
plt.yscale('log')
plt.xlabel("Time [Normalised]",fontsize =20)
plt.ylabel("First harmonic amplitude [Normalised]", fontsize =20)
plt.yticks(fontsize =20)
plt.xticks(fontsize=20)
plt.savefig('/home/dos515/Documents/Comp_Labs/Run/peak_finding.jpg')
plt.show()
#%% - PLOT FIT
plt.figure(figsize =(12,7),dpi=1000)
plt.plot(x_time[0], y_fh[0],'k',linewidth =3)
plt.plot(peak_positions[0], peak_heights[0], 'rx', markersize = 20)
plt.plot(peak_positions[0][0:4], exp_fit, 'r',linewidth =3)
plt.xlabel("Time [Normalised]",fontsize =20)
plt.ylabel("First harmonic amplitude [Normalised]", fontsize =20)
plt.yticks(fontsize =20)
plt.xticks(fontsize=20)
#plt.savefig('/home/dos515/Documents/Comp_Labs/Run/fit.jpg')
plt.show()
#%% - DENSITY CHECK

from epc1d_v1 import *
import numpy as np

ncells = 100  # Number of cells
L = 4*pi  # Length of the domain
my_positions = np.linspace(0,L,20)  # Array of positions
# Calculate the charge density based on positions
density_old = calc_density_old(my_positions, ncells, L)
density_new = calc_density(my_positions, ncells, L)  

#Check for difference in value between two density arrays
print(np.array_equal(density_old, density_new))


#%%
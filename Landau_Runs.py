#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:58:03 2023

@author: dos515
"""

from epc1d_v1_Landau import *

import matplotlib.pyplot as plt
import numpy as np
import time

#%% - RUNNING code and printing simulation time (to compare optimization)

num_particles = [1E3] #[1E3, 1E4, 1E5, 1E6, 1E7]

for i in range(len(num_particles)):
    print('SIMULATION', i)
    tic = time.time() #start clock
    
    L = 4.*pi
    npart = int(num_particles[i])
    ncells = 10
    
    pos, vel = landau(npart, L)
    s = Summary()
    run(pos, vel, L, ncells, [s], linspace(0.,20,50))
    
    toc = time.time() #end clock

    run_time = toc-tic #measuring time for each run

    print('Run time of simulation: ', run_time)
    print('-----------------------------------')
    
    
    
    

#%% - RUNNING code and saving data to csv (varying nparticle)
num_particles = [int(1e3),int(1e4),int(1e5),int(1e6), int(1e7)] #[1000,5000,10000,15000,20000]
num_cells = [20] # [10,50,100,500,1000]

for j in range(len(num_cells)):
    for i in range(len(num_particles)):
        tic = time.time() #start clock
        
        L = 4.*pi
        npart = num_particles[i]
        ncells = num_cells[j]
        
        
        pos, vel = landau(npart, L)
        s = Summary()
        run(pos, vel, L, ncells, [s], linspace(0.,20,50))
        
        toc = time.time() #clock end
        
        times = toc-tic #measuring time for each run
        
        savetxt("Run/data_np%s_nc%s.csv"%(npart,ncells), [np.array(s.t),np.array(s.firstharmonic),np.array([times]*len(s.t))], delimiter = ",")
        
        # plt.figure()
        # plt.plot(s.t, s.firstharmonic, label = 'npart = %s,ncells = %s'%(npart,ncells))
        # plt.yscale('log')
        # plt.xlabel("Time [Normalised]")
        # plt.ylabel("First harmonic amplitude [Normalised]")
        # plt.legend()
        # #plt.savefig('runs/Amp_np%s_nc%s.png'%(npart,ncells))
        #plt.show()
        
#%% - RUNNING code and saving data to csv (varying nparticle-extra)

num_particles = [int(2e5), int(5e5), int(8e5)]
num_cells = 20

for i in range(len(num_particles)):
    tic = time.time() #start clock
    
    L = 4.*pi
    npart = num_particles[i]
    ncells = num_cells
    
    
    pos, vel = landau(npart, L)
    s = Summary()
    run(pos, vel, L, ncells, [s], linspace(0.,20,50))
    
    toc = time.time() #clock end
    
    times = toc-tic #measuring time for each run
    
    savetxt("Run/data_np%s_nc%s.csv"%(npart,ncells), [np.array(s.t),np.array(s.firstharmonic),np.array([times]*len(s.t))], delimiter = ",")
    
#%% - RUNNING code and saving data to csv (varying ncell)
num_particles = 2E5
num_cells = [10,20,50,100,1000]

for j in range(len(num_cells)):
    tic = time.time() #start clock
    
    L = 4.*pi
    npart = int(num_particles)
    ncells = num_cells[j]
    
    
    pos, vel = landau(npart, L)
    s = Summary()
    run(pos, vel, L, ncells, [s], linspace(0.,20,50))
    
    toc = time.time() #clock end
    
    times = toc-tic #measuring time for each run
    
    savetxt("Run/data_np%s_nc%s.csv"%(npart,ncells), [np.array(s.t),np.array(s.firstharmonic),np.array([times]*len(s.t))], delimiter = ",")

#%% - RUNNING code and saving data to csv (varying length)
num_particles = int(2E5)
num_cells = 20
length_factor = [2,4,8,12,16]

for j in range(len(length_factor)):
    tic = time.time() #start clock
    
    L = length_factor[j]*pi
    npart = num_particles
    ncells = num_cells
    
    
    pos, vel = landau(npart, L)
    s = Summary()
    run(pos, vel, L, ncells, [s], linspace(0.,20,50))
    
    toc = time.time() #clock end
    
    times = toc-tic #measuring time for each run
    print('Done: ', length_factor[j])
    
    savetxt("Run/data_L%s.csv"%(length_factor[j]), [np.array(s.t),np.array(s.firstharmonic),np.array([times]*len(s.t))], delimiter = ",")
    
#%% - EXTRAS


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:53:12 2023

@author: dos515
"""
from epc1d_v1_TwoStream import *

import matplotlib.pyplot as plt
import numpy as np


#%% - RUNNING code and saving data to csv (varying nparticle)
v_set = [3,4,6,8,10,12]

for i in range(len(v_set)):  
    L = 100
    npart = 10000
    ncells = 20
    v = v_set[i]
    pos, vel = twostream(npart, L,v)
    s = Summary()
    run(pos, vel, L, ncells, [s], linspace(0.,20,50))
    print('-----------')
    print('Run %s DONE'%(i))
    savetxt("Run2/data_v%s.csv"%(v), [np.array(s.t),np.array(s.firstharmonic)], delimiter = ",")
    print('File %s SAVED'%(i))
    
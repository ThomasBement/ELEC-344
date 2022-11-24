# ---------------------------------------- #
# analysis_lab_4 [Python File]
# Written By: Thomas Bement
# Created On: 2022-11-19
# ---------------------------------------- #

"""
IMPORTS
"""
from cProfile import label
import cmath
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt

from math import floor
from tabulate import tabulate
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import make_interp_spline

"""
CONSTANTS
"""
in_path = './DATA/CSV/'
file_type = ''
thresh = {'V': 0.05, 'C': 0.01}
# CSV HEADERS:
#   Time (ms)	Ch1 V	Ch2 V	Ch3 V	Ch1 A	Ch2 A	Ch3 A	RPM	Torque
image_save = './IMG/'

# Equivalent circuit parameters
R1 = 0.208
X1 = 0.270
Xm = 2.979   
X2P = 0.270
R2P = 0.126

"""
FUNCTIONS
"""
def get_cmap(n, name='plasma'):
    return plt.cm.get_cmap(name, n)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def R2(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def get_file(search_path): 
    files = os.listdir(search_path) 
    print('Files in %s:' %(search_path)) 
    data = [] 
    for i in range(len(files)): 
        data.append([files[i], '%i' %(i)]) 
    print(tabulate(data, headers=['File Name', 'Index']), '\n') 
    idx = input('Enter index of file name to be used: ')
    idx_lis = [int(i) for i in idx.split(',') if i.isdigit()]
    file_lis = [files[i] for i in idx_lis]
    return file_lis 

def read_data(file_paths, delim):
    ans = {}
    for file_path in file_paths:
        ans[file_path] = {}
        headers = []
        R = open(in_path + file_path + file_type, 'r')
        R.seek(0)
        for line in itertools.islice(R, 0, 1):
            lineLis = line.split(delim)
            for i in range(len(lineLis)):
                if ('\n' in lineLis[i]) and (lineLis[i] != '\n'):
                    headers.append(lineLis[i].replace('\n', ''))
                    ans[file_path][lineLis[i].replace('\n', '')] = []
                elif (lineLis[i] != '') and (lineLis[i] != '\n'):
                    headers.append(lineLis[i])
                    ans[file_path][lineLis[i]] = []  
        R.seek(0)
        for line in itertools.islice(R, 1, None):
            lineLis = line.split(delim)
            for i in range(len(headers)):
                ans[file_path][headers[i]].append(float(lineLis[i]))
    return ans

def slip(w_syn, w_r):
    return (w_syn-w_r)/w_syn

def Te(p, we, wr, V1, R1, R2P, X1, X2P):
    S = slip(2*we/p, wr)
    return 3*(p/2)*(V1**2/we)*((R2P/S)/((R1+R2P/S)**2+(X1+X2P)**2))


def task1b(dat_in):
    TIME = []
    V = {'Ch1 V': [], 'Ch2 V': [], 'Ch3 V': []}
    A = {'Ch1 A': [], 'Ch2 A': [], 'Ch3 A': []}
    RPM = []
    name_use = []
    for name in dat_in:
        if ('1B' in name):
            name_use.append(name)
            TIME = np.array(dat_in[name]['Time (ms)']) 
            V['Ch1 V'] = np.array(dat_in[name]['Ch1 V'])
            A['Ch1 A'] = np.array(dat_in[name]['Ch1 A'])
            RPM = np.array(dat_in[name]['RPM'])

    #--- Plotting ---#
    idx_start = find_nearest(TIME, 1400)
    idx_end = find_nearest(TIME, 1800)
    cmap = get_cmap(8, 'plasma')   
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(TIME[idx_start:idx_end], V['Ch1 V'][idx_start:idx_end], label='CH1_V', color=cmap(0))
    ax2.plot(TIME[idx_start:idx_end], A['Ch1 A'][idx_start:idx_end], label='CH1_A', color=cmap(3))
    
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('CH1 Current [A]')
    ax2.set_ylabel('CH1 Voltage [V]')

    
    ax1.legend(loc='upper center', bbox_to_anchor=(0, -0.05), fancybox=True, shadow=True)
    ax2.legend(loc='upper center', bbox_to_anchor=(1, -0.05), fancybox=True, shadow=True)

    plt.savefig('%s/Task_1B_TS.png' %(image_save), bbox_inches='tight')
    plt.show()
    plt.close()

def task1c(dat_in):
    TIME = []
    V = {'Ch1 V': [], 'Ch2 V': [], 'Ch3 V': []}
    A = {'Ch1 A': [], 'Ch2 A': [], 'Ch3 A': []}
    RPM = []
    name_use = []
    for name in dat_in:
        if ('1C' in name):
            name_use.append(name)
            TIME = np.array(dat_in[name]['Time (ms)']) 
            V['Ch1 V'] = np.array(dat_in[name]['Ch1 V'])
            A['Ch1 A'] = np.array(dat_in[name]['Ch1 A'])
            RPM = np.array(dat_in[name]['RPM'])

    #--- Plotting ---#
    idx_start = find_nearest(TIME, 700)
    idx_end = find_nearest(TIME, 1100)
    cmap = get_cmap(8, 'plasma')   
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(TIME[idx_start:idx_end], V['Ch1 V'][idx_start:idx_end], label='CH1_V', color=cmap(0))
    ax2.plot(TIME[idx_start:idx_end], A['Ch1 A'][idx_start:idx_end], label='CH1_A', color=cmap(3))
    
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('CH1 Current [A]')
    ax2.set_ylabel('CH1 Voltage [V]')

    
    ax1.legend(loc='upper center', bbox_to_anchor=(0, -0.05), fancybox=True, shadow=True)
    ax2.legend(loc='upper center', bbox_to_anchor=(1, -0.05), fancybox=True, shadow=True)

    plt.savefig('%s/Task_1C_TS.png' %(image_save), bbox_inches='tight')
    plt.show()
    plt.close()

def task5():
    V1 = 19
    # Measured Values
    PF_ang_m = np.array([-73, -70.4, -65.5, -60.2, -56.6, -53.4])
    PF_m = np.cos(PF_ang_m*np.pi/180)
    n_m = np.array([1792, 1789, 1782, 1775, 1768, 1761])
    wr_m = n_m/60*2*np.pi
    torque_m = np.array([0.280000, 0.370000, 0.560000, 0.730000, 0.900000, 1.070000])
    P_in_m = np.array([98.4, 115, 149, 182.1, 216.8, 249.8])
    P_out_m = wr_m*torque_m
    S_m = slip(4*np.pi*60/4, wr_m)
    eff_m = P_out_m/P_in_m*100
    i_ph = 5.97
    z_m = []
    for i in range(len(S_m)):
        z_m.append(complex(R1,X1) + complex(0,Xm)*complex(R2P/S_m[i],X2P)/(complex(R2P/S_m[i],Xm+X2P)))
    z_m = np.array(z_m)

    # Equivalent Circuit Values
    wr = np.linspace(0, 1800/60*2*np.pi, 256)
    S = slip(4*np.pi*60/4, wr)
    n = wr*60/(2*np.pi)
    torque = Te(4, 2*np.pi*60, wr, V1, R1, R2P, X1, X2P)
    cmap = get_cmap(4, 'plasma')
    z_t = []
    for i in range(len(S)):
        z_t.append(complex(R1,X1) + complex(0,Xm)*complex(R2P/S[i],X2P)/(complex(R2P/S[i],Xm+X2P)))
    z_t = np.array(z_t)
    I1 = np.absolute(V1/z_t)
    #P_in = 3*V1*abs(I1)*PF
    P_out = wr*torque
    P_loss = 3*(np.sqrt(2)*i_ph)**2*R1
    eff = (1-(P_loss/P_in_m))*100
    I1_m = V1/np.absolute(z_m)
    PF = P_in_m/(3*V1*I1_m)

    
    # 5b1
    diff = np.nan*np.ones_like(n_m)
    for i in range(len(n_m)):
        idx = find_nearest(n, n_m[i])
        diff[i] = abs((torque_m[i]-torque[idx])/torque[idx])*100
    print('5A - Percent Error [%]:')
    print(diff)
    plt.plot(n, torque, label='Theoretical Characteristic', color=cmap(0))
    plt.plot(n_m, torque_m, marker='o', label='Measured Characteristic', color=cmap(1))
    plt.xlabel('Rotational Speed [RPM]')
    plt.ylabel('Torque [Nm]')
    plt.legend()
    plt.savefig('%s/Task_5B1_Wide.png' %(image_save), bbox_inches='tight')
    plt.xlim([1700, 1800])
    plt.savefig('%s/Task_5B1_Narrow.png' %(image_save), bbox_inches='tight')
    plt.close()

    # 5b2
    X_Y_Spline = make_interp_spline(torque_m, eff)
    # Returns evenly spaced numbers
    # over a specified interval.
    torque_m_refine = np.linspace(min(torque_m), max(torque_m), 256)
    eff = X_Y_Spline(torque_m_refine)
    plt.plot(torque_m_refine, eff, label='Theoretical Efficiency', color=cmap(0))
    plt.plot(torque_m, eff_m, marker='o', label='Measured Efficiency', color=cmap(1))
    plt.xlabel('Torque [Nm]')
    plt.ylabel('Efficiency [%]')
    plt.legend()
    plt.savefig('%s/Task_5B2.png' %(image_save), bbox_inches='tight')
    plt.show()
    plt.close()

    # 5b3
    plt.plot(torque_m, PF, label='Theoretical PF', color=cmap(0))
    plt.plot(torque_m, PF_m, marker='o', label='Measured PF', color=cmap(1))
    plt.xlabel('Torque [Nm]')
    plt.ylabel('PF')
    plt.legend()
    plt.savefig('%s/Task_5B3.png' %(image_save), bbox_inches='tight')
    #plt.xlim([1700, 1800])
    plt.show()
    plt.close()

    # 5b4
    V1_max = 19.06
    V1_min = 13.08
    # Measured Values
    torque_m_max = 1.07
    n_m_max = 1761
    wr_m_max = n_m_max/60*2*np.pi
    torque_m_min = 1.04
    n_m_min = 1696
    wr_m_max = n_m_min/60*2*np.pi

    # Equivalent Circuit Values
    wr = np.linspace(0, 1800/60*2*np.pi, 256)
    S = slip(4*np.pi*60/4, wr)
    n = wr*60/(2*np.pi)
    torque_max = Te(4, 2*np.pi*60, wr, V1_max, R1, R2P, X1, X2P)
    torque_min = Te(4, 2*np.pi*60, wr, V1_min, R1, R2P, X1, X2P)
    cmap = get_cmap(6, 'plasma')
        
    plt.plot(n, torque_max, label='Theoretical Characteristic V1 = %.2f [V]' %(V1_max), color=cmap(0))
    plt.plot(n, torque_min, label='Theoretical Characteristic V1 = %.2f [V]' %(V1_min), color=cmap(1))
    plt.scatter(n_m_max, torque_m_max, label='Measured Characteristic V1 = %.2f [V]' %(V1_max), color=cmap(2))
    plt.scatter(n_m_min, torque_m_min, label='Measured Characteristic V1 = %.2f [V]' %(V1_min), color=cmap(3))
    plt.xlabel('Rotational Speed [RPM]')
    plt.ylabel('Torque [Nm]')
    plt.legend()
    plt.savefig('%s/Task_5B4_Wide.png' %(image_save), bbox_inches='tight')
    plt.xlim([1600, 1800])
    plt.savefig('%s/Task_5B4_Narrow.png' %(image_save), bbox_inches='tight')
    plt.close()

"""
MAIN
"""
files = os.listdir(in_path) 
dat = read_data(files, '\t')
#task1b(dat)
#task1c(dat)
task5()
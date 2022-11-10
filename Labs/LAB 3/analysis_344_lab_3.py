# ---------------------------------------- #
# analysis_344_lab_3 [Python File]
# WRitten By: Thomas Bement
# Created On: 2022-11-03
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

"""
CONSTANTS
"""
in_path = './DATA/CSV/'
file_type = ''
thresh = {'V': 0.05, 'C': 0.01}
#           0               1           2           3           4           5           6       #
idx_k = [   'Time (ms)',    'Ch1 V',    'Ch2 V',    'Ch1 A',    'Ch2 A',    'RPM',      'Torque']
image_save = './IMG/'
ra = 0.61

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

def plot_keys(dat, key_x, keys_y):
    for key in keys_y:
        plt.plot(dat[key_x], dat[key])

def Tf(WR, a, b, c):
    return (a*WR**2)+(b*WR)+(c)

def Tm(WR, a, b):
    return (a*WR)+(b)

def task1b(dat):
    V1 = []
    I1 = []
    RPM = []
    for name in dat:
        if 'Task1B' in name:
            V1.append(np.mean(dat[name]['Ch1 V']))
            I1.append(np.mean(dat[name]['Ch1 A']))
            RPM.append(np.mean(dat[name]['RPM']))
    WR = [x*2*np.pi/60 for x in RPM]
    kt = []
    for i in range(len(V1)):
        kt.append((V1[i]-I1[i]*ra)/WR[i])
    kt_ave = np.mean(np.array(kt))
    torque_fric = [x*kt_ave for x in I1]

    #--- Fitting ---#
    p0 = [-1, 1, 1]
    popt, _ = curve_fit(Tf, np.array(WR), np.array(torque_fric), p0=p0)
    x_fit = np.linspace(min(WR), max(WR), 256)
    y_fit = Tf(x_fit, *popt)
    print('Task 1B Fit R^2: %f' %(R2(np.array(torque_fric), Tf(np.array(WR), *popt))))
    #--- Plotting ---#
    WR, torque_fric = zip(*sorted(zip(WR, torque_fric)))
    plt.scatter(WR, torque_fric)
    plt.plot(x_fit, y_fit, alpha=0.6)
    plt.title('Task 1-B Friction Speed Curve:')
    plt.xlabel('Rotational Speed [rad/s]')
    plt.ylabel('Friction Torque [N*m]')
    plt.savefig('%s/Task_1B_Torque_Speed.png' %(image_save), bbox_inches='tight')
    plt.show()
    plt.close()

    return popt, kt_ave

def task1c(dat_in, popt_Ft, kt):
    for name in dat_in:
        if 'Task1C' in name:
            dat = dat_in[name]
    TIME = dat['Time (ms)'] 
    CH1_V = dat['Ch1 V']
    CH2_V = dat['Ch2 V']
    CH1_A = dat['Ch1 A']
    CH2_A = dat['Ch2 A']
    EA = np.array(CH1_V)-np.array(CH1_A)*ra
    RPM = dat['RPM']
    WR = [x*2*np.pi/60 for x in RPM]
    TRQ = dat['Torque']

    #--- Integreation ---#
    T_start = 170   # [ms]
    T_stop = 800    # [ms]
    idx_start = find_nearest(TIME, T_start) 
    idx_stop =  find_nearest(TIME, T_stop)
    Tf_int = 0
    for i in range(idx_start, idx_stop):
        dt = TIME[i]-TIME[i-1]
        Tf_int += Tf(WR[i], *popt_Ft)*dt
    denom_1 = WR[idx_stop] - WR[idx_start]
    denom_2 = (EA[idx_stop] - EA[idx_start])/kt
    J_rotor_1 = abs(Tf_int/denom_1/1000)
    J_rotor_2 = abs(Tf_int/denom_2/1000)
    print('Integral Form:')
    print('Task 1C J-Rotor Using W_r: %f' %(J_rotor_1))
    print('Task 1C J-Rotor Using E_a: %f' %(J_rotor_2))

    #--- Differential ---#
    idx_ave = int((idx_start+idx_stop)/2)
    J_rotor_1 = abs(Tf(WR[idx_ave], *popt_Ft)*(TIME[idx_stop]-TIME[idx_start])/(WR[idx_stop]-WR[idx_start])/1000)
    J_rotor_2 = abs(Tf(WR[idx_ave], *popt_Ft)*(TIME[idx_stop]-TIME[idx_start])*kt/(EA[idx_stop]-EA[idx_start])/1000)
    print('Differential Form:')
    print('Task 1C J-Rotor Using W_r: %f' %(J_rotor_1))
    print('Task 1C J-Rotor Using E_a: %f' %(J_rotor_2))

    #--- Plotting ---#
    cmap = get_cmap(3, 'plasma')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(TIME[idx_start:idx_stop], WR[idx_start:idx_stop], label='WR', color=cmap(0))
    ax2.plot(TIME[idx_start:idx_stop], EA[idx_start:idx_stop], label='Ea', color=cmap(1))
    
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('WR [rad/s]')
    ax2.set_ylabel('Ea [V]')

    ax1.legend(loc=0)
    ax2.legend(loc=2)
    plt.savefig('%s/Task_1C_TS.png' %(image_save), bbox_inches='tight')
    plt.show()
    plt.close()

def task1d(dat_in, popt_Ft, kt):
    for name in dat_in:
        if 'Task1D' in name:
            dat = dat_in[name]
    TIME = dat['Time (ms)'] 
    CH1_V = dat['Ch1 V']
    CH1_A = dat['Ch1 A']
    EA = np.array(CH1_V)-np.array(CH1_A)*ra
    RPM = dat['RPM']
    WR = [x*2*np.pi/60 for x in RPM]

    #--- Integreation ---#
    T_start = 340   # [ms]
    T_stop = 1200    # [ms]
    idx_start = find_nearest(TIME, T_start) 
    idx_stop =  find_nearest(TIME, T_stop)
    Tf_int = 0
    for i in range(idx_start, idx_stop):
        dt = TIME[i]-TIME[i-1]
        Tf_int += Tf(WR[i], *popt_Ft)*dt
    denom_1 = WR[idx_stop] - WR[idx_start]
    denom_2 = (EA[idx_stop] - EA[idx_start])/kt
    J_rotor_1 = abs(2*Tf_int/denom_1/1000)
    J_rotor_2 = abs(2*Tf_int/denom_2/1000)
    print('Integral Form:')
    print('Task 1C J-Rotor Using W_r: %f' %(J_rotor_1))
    print('Task 1C J-Rotor Using E_a: %f' %(J_rotor_2))

    #--- Differential ---#
    idx_ave = int((idx_start+idx_stop)/2)
    J_rotor_1 = abs(2*Tf(WR[idx_ave], *popt_Ft)*(TIME[idx_stop]-TIME[idx_start])/(WR[idx_stop]-WR[idx_start])/1000)
    J_rotor_2 = abs(2*Tf(WR[idx_ave], *popt_Ft)*(TIME[idx_stop]-TIME[idx_start])*kt/(EA[idx_stop]-EA[idx_start])/1000)
    print('Differential Form:')
    print('Task 1C J-Rotor Using W_r: %f' %(J_rotor_1))
    print('Task 1C J-Rotor Using E_a: %f' %(J_rotor_2))

    #--- Plotting ---#
    cmap = get_cmap(3, 'plasma')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(TIME[idx_start:idx_stop], WR[idx_start:idx_stop], label='WR', color=cmap(0))
    ax2.plot(TIME[idx_start:idx_stop], EA[idx_start:idx_stop], label='Ea', color=cmap(1))
    
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('WR [rad/s]')
    ax2.set_ylabel('Ea [V]')

    ax1.legend(loc=0)
    ax2.legend(loc=2)
    plt.savefig('%s/Task_1D_TS.png' %(image_save), bbox_inches='tight')
    plt.show()
    plt.close()  

def task2b(dat):
    RPM = []
    TRQ = []
    CH1_A = []
    for name in dat:
        if 'Task2B' in name:
            RPM.append(np.mean(dat[name]['RPM']))
            TRQ.append(np.mean(dat[name]['Torque']))
            CH1_A.append(np.mean(dat[name]['Ch1 A']))
    WR = [x*2*np.pi/60 for x in RPM]
    TRQ = [-x for x in TRQ]

    #--- Fitting ---#
    p0 = [1, 0]
    popt, _ = curve_fit(Tm, np.array(WR), np.array(TRQ), p0=p0)
    x_fit = np.linspace(min(WR), max(WR), 256)
    y_fit = Tm(x_fit, *popt)
    print('Task 2B Fit R^2: %f' %(R2(np.array(TRQ), Tm(np.array(WR), *popt))))
    #--- Plotting ---#
    WR, TRQ = zip(*sorted(zip(WR, TRQ)))
    
    cmap = get_cmap(4, 'plasma')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.scatter(WR, TRQ, label='Original Data', color=cmap(0))
    ax1.plot(x_fit, y_fit, alpha=0.6, label='Fit Data', color=cmap(1))
    ax2.plot(WR, CH1_A, color=cmap(3))
    
    plt.title('Task 2-B Friction Speed Curve:')
    ax1.set_xlabel('WR [rad/s]')
    ax1.set_ylabel('Mechanical Torque [N*m]')
    ax2.set_ylabel('CH1 Current [A]')

    ax1.legend(loc=0)
    ax2.legend(loc=2)
    plt.savefig('%s/Task_2B_Torque_Speed.png' %(image_save), bbox_inches='tight')
    plt.show()
    plt.close()

    return popt

def task3a(dat_in, kv):
    TIME = []
    CH1_V = []
    CH1_A = []
    RPM = []
    name_use = []
    for name in dat_in:
        if ('Task3A' in name) and ('05' in name):
            name_use.append(name)
            TIME = np.array(dat_in[name]['Time (ms)']) 
            CH1_V = np.array(dat_in[name]['Ch1 V'])
            CH1_A = np.array(dat_in[name]['Ch1 A'])
            RPM = np.array(dat_in[name]['RPM'])

    #--- Peak Finding ---#    
    peaks, _ = find_peaks(CH1_A, height = 3.5, prominence=1, distance=100)
    valleys, _ = find_peaks(CH1_A*-1, height = -1, prominence=1, distance=100)
    idx_keep = min(len(peaks), len(valleys))
    peaks = peaks[:idx_keep]
    valleys = valleys[:idx_keep]

    ripple = []
    for i in range(len(peaks)):
        ripple.append(CH1_A[peaks[i]] - CH1_A[valleys[i]])
    
    print('Task 3A CH1_A Current Ripple Mean Value: %f [A]' %(np.mean(np.array(ripple))))
    print('Task 3A CH1_A Current Ripple STDEV: %f [A]' %(np.std(np.array(ripple))))

    #--- Calculating Inductance ---#
    WR = np.array([x*2*np.pi/60 for x in RPM])
    dt_di = []
    for i in range(len(peaks)):
        dt_di.append(abs((TIME[peaks[i]]-TIME[valleys[i]])/(CH1_A[peaks[i]]-CH1_A[valleys[i]])))
    LA = np.mean(np.array(dt_di))*(np.mean(CH1_V)-(kv*np.mean(WR)))
    print('Task 3A Inductance First: %f [H]' %(LA))
    LA = np.mean(np.array(dt_di))*(np.mean(CH1_V)-(kv*np.mean(WR)+ra*np.mean(CH1_A)))
    print('Task 3A Inductance Full: %f [H]' %(LA))
    print('DT: %f' %(np.mean(TIME[peaks]-TIME[valleys])))
    print('DI: %f' %(np.mean(CH1_A[peaks]-CH1_A[valleys])))
    print('VA: %f' %((np.mean(CH1_V))))
    print('KV: %f' %(kv))
    print('WR: %f' %(np.mean(WR)))
    print('IA: %f' %(np.mean(CH1_A)))
    print('RA: %f' %(ra))

    #--- Plotting ---#
    cmap = get_cmap(8, 'plasma')   
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(TIME, CH1_A, label='CH1_A', color=cmap(0))
    ax1.plot(TIME[peaks], CH1_A[peaks], "x", color=cmap(1))
    ax1.plot(TIME[valleys], CH1_A[valleys], "x", color=cmap(2))
    ax2.plot(TIME, CH1_V, label='CH1_V', color=cmap(3))
    
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('CH1 Current [A]')
    ax2.set_ylabel('CH1 Voltage [V]')

    ax1.legend(loc='upper center', bbox_to_anchor=(0, -0.05), fancybox=True, shadow=True)
    ax2.legend(loc='upper center', bbox_to_anchor=(1, -0.05), fancybox=True, shadow=True)

    plt.savefig('%s/Task_3A_TS.png' %(image_save), bbox_inches='tight')
    plt.show()
    plt.close()      

def task3b(dat_in, kv):
    TIME = []
    CH1_V = []
    CH1_A = []
    RPM = []
    name_use = []
    for name in dat_in:
        if ('Task3B' in name) and ('05' in name):
            name_use.append(name)
            TIME = np.array(dat_in[name]['Time (ms)']) 
            CH1_V = np.array(dat_in[name]['Ch1 V'])
            CH1_A = np.array(dat_in[name]['Ch1 A'])
            RPM = np.array(dat_in[name]['RPM'])

    #--- Peak Finding ---#    
    peaks, _ = find_peaks(CH1_A, height = 2.6, prominence=1, distance=10)
    valleys, _ = find_peaks(CH1_A*-1, height = -1.4, prominence=1, distance=10)
    idx_keep = min(len(peaks), len(valleys))
    peaks = peaks[:idx_keep]
    valleys = valleys[:idx_keep]

    ripple = []
    for i in range(len(peaks)):
        ripple.append(CH1_A[peaks[i]] - CH1_A[valleys[i]])
    
    print('Task 3B CH1_A Current Ripple Mean Value: %f [A]' %(np.mean(np.array(ripple))))
    print('Task 3B CH1_A Current Ripple STDEV: %f [A]' %(np.std(np.array(ripple))))

    #--- Calculating Inductance ---#
    WR = np.array([x*2*np.pi/60 for x in RPM])
    dt_di = []
    for i in range(len(peaks)):
        dt_di.append(abs((TIME[peaks[i]]-TIME[valleys[i]])/(CH1_A[peaks[i]]-CH1_A[valleys[i]])))
    LA = np.mean(np.array(dt_di))*(np.mean(CH1_V)-(kv*np.mean(WR)))
    print('Task 3A Inductance First: %f [H]' %(LA))
    LA = np.mean(np.array(dt_di))*(np.mean(CH1_V)-(kv*np.mean(WR)+ra*np.mean(CH1_A)))
    print('Task 3B Inductance Full: %f [H]' %(LA))
    print('DT: %f' %(np.mean(TIME[peaks]-TIME[valleys])))
    print('DI: %f' %(np.mean(CH1_A[peaks]-CH1_A[valleys])))
    print('VA: %f' %((np.mean(CH1_V))))
    print('KV: %f' %(kv))
    print('WR: %f' %(np.mean(WR)))
    print('IA: %f' %(np.mean(CH1_A)))
    print('RA: %f' %(ra))

    #--- Plotting ---#
    cmap = get_cmap(8, 'plasma')   
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(TIME, CH1_A, label='CH1_A', color=cmap(0))
    ax1.plot(TIME[peaks], CH1_A[peaks], "x", color=cmap(1))
    ax1.plot(TIME[valleys], CH1_A[valleys], "x", color=cmap(2))
    ax2.plot(TIME, CH1_V, label='CH1_V', color=cmap(3))
    
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('CH1 Current [A]')
    ax2.set_ylabel('CH1 Voltage [V]')

    
    ax1.legend(loc='upper center', bbox_to_anchor=(0, -0.05), fancybox=True, shadow=True)
    ax2.legend(loc='upper center', bbox_to_anchor=(1, -0.05), fancybox=True, shadow=True)

    plt.savefig('%s/Task_3B_TS.png' %(image_save), bbox_inches='tight')
    plt.show()
    plt.close()

"""
MAIN
"""
files = os.listdir(in_path) 
dat = read_data(files, '\t')

Ft_coeff, kt = task1b(dat)
print(Ft_coeff)
task1c(dat, Ft_coeff, kt)
task1d(dat, Ft_coeff, kt)
Fm_coeff = task2b(dat)
print(Fm_coeff)
task3a(dat, kt)
task3b(dat, kt)
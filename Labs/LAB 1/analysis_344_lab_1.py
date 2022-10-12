# ---------------------------------------- #
# analysis_344_lab_1 [Python File]
# Written By: Thomas Bement
# Created On: 2022-10-05
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

"""
CONSTANTS
"""
in_path = './CSV_INPUT/'
file_type = ''
thresh = {'V': 0.05, 'C': 0.01}
f = 60
w = f*(2*np.pi)/1000 # 1000 for time in ms
#           0               1           2           3           4           5           6       #
idx_k = [   'Time (ms)',    'Ch1 V',    'Ch2 V',    'Ch3 V',    'Ch1 A',    'Ch2 A',    'Ch3 A']
image_save = './IMG/'

"""
FUNCTIONS
"""
def get_file(search_path): 
    files = os.listdir(search_path) 
    print('Files in %s:' %(search_path)) 
    data = [] 
    for i in range(len(files)): 
        data.append([files[i], '%i' %(i)]) 
    print(tabulate(data, headers=['File Name', 'Index']), '\n') 
    idx = int(input('Enter index of file name to be used: ')) 
    return files[idx] 

def signal(t, a, b):
    return a*np.sin(w*t + b)

def fit_signal(x_dat, y_dat, sig_type):
    n_points = len(x_dat)
    if (max(y_dat) <= thresh[sig_type]):
        return [np.zeros(n_points), np.zeros(n_points), [0, 0]]
    else:
        p0 = [max(y_dat), 0]
        popt, _ = curve_fit(signal, x_dat, y_dat, p0=p0)
        x_fit = np.linspace(x_dat[0], x_dat[-1], n_points)
        y_fit = signal(x_fit, *popt)
    return [x_fit, y_fit, popt]

def read_data(file_path, delim):
    print('Reading File: %s' %(file_path))
    ans = {}
    headers = []
    R = open(file_path, 'r')
    R.seek(0)
    for line in itertools.islice(R, 0, 1):
        lineLis = line.split(delim)
        for i in range(len(lineLis)):
            if ('\n' in lineLis[i]) and (lineLis[i] != '\n'):
                headers.append(lineLis[i].replace('\n', ''))
                ans[lineLis[i].replace('\n', '')] = []
            elif (lineLis[i] != '') and (lineLis[i] != '\n'):
                headers.append(lineLis[i])
                ans[lineLis[i]] = []  
    R.seek(0)
    for line in itertools.islice(R, 1, None):
        lineLis = line.split(delim)
        for i in range(len(headers)):
            ans[headers[i]].append(float(lineLis[i]))
    for key in ans:
        ans[key] = np.array(ans[key])
    time = ans['Time (ms)']
    ans_fit = {}
    for key in ans:
        if (key == 'Time (ms)'):
            ans_fit[key] = ans[key]
        else:
            if ('V' in key):
                ans_fit[key] = fit_signal(time, ans[key], 'V')[1]
            elif ('A' in key):
                ans_fit[key] = fit_signal(time, ans[key], 'C')[1]
            else:
                print('Error: %s' %(key))
    print('Done\n')
    return ans, ans_fit

def calc_phas(time, signal):
    t_max = 1000*(floor((time[-1]/1000)*f)/f)
    i_max = 1 + np.searchsorted(time, t_max)
    
    # trapezoid sum:
    t = time[0:i_max]
    s_re = signal[0:i_max]*np.cos(w*t)
    p_re = ((t[1:] - t[:-1])*(s_re[1:] + s_re[:-1])).sum()*(1/t[-1])/np.sqrt(2)
    s_im = signal[0:i_max]*np.sin(w*t)
    p_im = ((t[1:] - t[:-1])*(s_im[1:] + s_im[:-1])).sum()*(-1/t[-1])/np.sqrt(2)
    return complex(p_re, p_im)

def plot_phas(phas_lis, name):
    f, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
    ax1.set_title('Voltage [V]')
    ax2.set_title('Current [A]')
    ax1.set_rlabel_position(180)
    ax2.set_rlabel_position(180)
    ax1.yaxis.get_major_locator().base.set_params(nbins=4)
    ax2.yaxis.get_major_locator().base.set_params(nbins=4)
    for i in range(len(phas_lis)):
        re = phas_lis[i][0].real
        im = phas_lis[i][0].imag
        r = abs(phas_lis[i][0])
        theta = np.angle(phas_lis[i][0])
        if ('V' in phas_lis[i][1]):
            ax1.plot([0, theta], [0, r], label = phas_lis[i][1])
        elif ('A' in phas_lis[i][1]):
            ax2.plot([0, theta], [0, r], label = phas_lis[i][1])
    #plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    f.legend(loc='lower center', fancybox=True, shadow=True, ncol=len(phas_lis))
    f.tight_layout(pad=1.0)
    plt.savefig('%s%s_Phas.png' %(image_save, name), bbox_inches='tight')
    #plt.show()
    #plt.close()

def gen_phas(time, dat, keys):
    ans = []
    for key in keys:
        temp = (calc_phas(time, dat[key]), key)
        ans.append(temp)
    return ans

def power(time, v_t, i_t, name):
    P_inst = v_t*i_t
    v_phas = calc_phas(time, v_t)
    i_phas = calc_phas(time, i_t)
    #P_real = ave(P_inst)/cycle
    PF_ang = np.arctan(v_phas.real/v_phas.imag)-np.arctan(i_phas.real/i_phas.imag)
    P_real = abs(v_phas)*abs(i_phas)*np.cos(PF_ang)
    P_react = abs(v_phas)*abs(i_phas)*np.sin(PF_ang)
    P_app = np.sqrt(P_real**2 + P_react**2)
    P_dat = {'PF Angle': PF_ang*180/np.pi, 'P Real': P_real, 'Q': P_react, 'S': P_app} 
    table = []
    for key in P_dat:
        table.append([])
        table[-1].append(key)
        table[-1].append(P_dat[key])
    print(tabulate(table), '\n')
    plt.plot(time, v_t, label = 'Voltage [V]')
    plt.plot(time, i_t, label = 'Current [A]')
    plt.plot(time, P_inst, label = 'Instantaneous Power [W]')
    plt.plot(time, np.ones_like(time)*P_real, label = 'Real Power [W]')
    plt.plot(time, np.ones_like(time)*P_react, label = 'Reactive Power [W]')
    plt.xlabel('Time [ms]')
    plt.ylabel('Signal Magnitude')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.savefig('%s%s_Pow.png' %(image_save, name), bbox_inches='tight')
    #plt.show()
    #plt.close()

def circuit_calcs(dat, name):
    if '1-A' in name:
        print(name + ': 1-A')
    elif '1-B' in name:
        print(name + ': 1-B')
    elif '1-C' in name:
        print('Results for %s' %(name))
        power(dat['Time (ms)'], dat['Ch1 V'], dat['Ch1 A'], name)
        phas_lis_1C = gen_phas(dat['Time (ms)'], dat, ['Ch1 V',    'Ch2 V',    'Ch3 V',    'Ch1 A',    'Ch2 A',    'Ch3 A'])
        plot_phas(phas_lis_1C, name)
    elif '1-D' in name:
        print('Results for %s' %(name))
        phas_lis_1D = gen_phas(dat['Time (ms)'], dat, ['Ch1 V',    'Ch2 V',    'Ch3 V',    'Ch1 A',    'Ch2 A',    'Ch3 A'])
        plot_phas(phas_lis_1D, name)
    elif '2-A' in name:
        print('Results for %s' %(name))
        phas_lis_2A = gen_phas(dat['Time (ms)'], dat, ['Ch1 V',    'Ch2 V',    'Ch3 V',    'Ch1 A',    'Ch2 A',    'Ch3 A'])
        plot_phas(phas_lis_2A, name)
    elif '2-B' in name:
        print(name + ': 2-B')
    elif '' in name:
        i = 0
    else:
        print("%s not recognised as valid name..." %(name))

def plot_2d(dat):
    time = 1
    voltage = []
    current = []
    for i, key in enumerate(dat):
        if ('V' in key):
            temp = (dat[key], key)
            voltage.append(temp)
        elif ('A' in key):
            temp = (dat[key], key)
            current.append(temp)
        elif (('Time (ms)' in key) or ('time (ms)' in key)):
            time = dat[key]
        else:
            print(key)
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Voltage [V]')
    ax2.set_title('Current [A]')
    for i in range(len(voltage)):
        fit_dat = fit_signal(time, voltage[i][0], 'V')
        ax1.plot(fit_dat[0], fit_dat[1], label = '%s Fit' %(voltage[i][1]))
        ax1.plot(time, voltage[i][0], label = '%s Real' %(voltage[i][1]))
    for i in range(len(current)):
        fit_dat = fit_signal(time, current[i][0], 'C')
        ax2.plot(fit_dat[0], fit_dat[1], label = '%s Fit' %(current[i][1]))
        ax2.plot(time, current[i][0], label = '%s Real' %(current[i][1]))
    f.legend(loc='lower center', fancybox=True, shadow=True, ncol=2*len(voltage))
    plt.show()
    plt.close()

def harmonics(time, signal):
    fy = np.fft.fft(signal)
    N = len(time)
    T = (time[-1] - time[0]) * (1 + 1.0/N) / 1000
    freqs = (np.arange(0, len(fy)) / T)
    abs_fy = abs(fy)
    maxamp = max(abs_fy)
    plt.plot(freqs, abs_fy/maxamp)
    plt.xlim(0, 1000)
    plt.ylim(0, 0.2)
    plt.title('Voltage Supply Harmonics')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude Normalized by Fundamental Component [#]')
    plt.savefig('%sHarmonics.png' %(image_save), bbox_inches='tight')
    plt.show()
    
"""
MAIN
"""
file_name = get_file(in_path)
dat, fit_dat = read_data(in_path + file_name + file_type, '\t')
circuit_calcs(dat, file_name)
#harmonics(dat['Time (ms)'], dat['Ch1 V'])
#plot_2d(dat)
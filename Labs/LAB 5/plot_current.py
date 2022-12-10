# ---------------------------------------- #
# plot_current [Python File]
# Written By: Thomas Bement
# Created On: 2022-12-07
# ---------------------------------------- #

"""
IMPORTS
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
CONSTANTS
"""
data_path = './DATA/CSV'
img_path = './DATA/IMG'
"""
FUNCTIONS
"""
def plot_current(dat, name, time_offset=0):
    time = np.array(dat['Time (ms)'])+time_offset
    current = np.array(dat['Ch1 A'])
    plt.plot(time, current, label=name)

def plot_voltage(dat, name, time_offset=0):
    time = np.array(dat['Time (ms)'])+time_offset
    voltage = np.array(dat['Ch1 V'])
    plt.plot(time, voltage, label=name)

"""
MAIN
"""
dat_4A = pd.read_csv('%s/Task4A_M7_data' %(data_path), delimiter='\t')
dat_4B = pd.read_csv('%s/Task4C' %(data_path), delimiter='\t')

plot_voltage(dat_4A, 'Task 4A')
plot_voltage(dat_4B, 'Task 4B', 0.158)
plt.xlabel('Time [ms]')
plt.ylabel('Voltage [V]')
plt.legend()
#plt.xlim([2.5, 10])
plt.savefig('%s/Wide_View' %(img_path), bbox_inches='tight')
plt.xlim([3, 6])
plt.ylim([0, -30])
plt.savefig('%s/Zoom_View' %(img_path), bbox_inches='tight')
plt.show()
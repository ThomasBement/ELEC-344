# ---------------------------------------- #
# problem_1_d [Python File]
# Written By: Thomas Bement
# Created On: 2022-10-14
# ---------------------------------------- #

"""
IMPORTS
"""
import cmath

import numpy as np
import matplotlib.pyplot as plt

"""
FUNCTIONS
"""
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

"""
MAIN
"""
N1 = 200.0
N2 = 100.0
a = N1/N2

# Impedances
r_load_s = np.linspace(0, 200, 2**14) #complex(10, 0)
r_load_p = r_load_s*(a**2)

r_winding = complex(0.5, 0)
x_winding = complex(0, 1.94)

r_core = complex(480, 0)
x_core = complex(0, 123.94)

z_1 = r_winding + x_winding
z_2 = 1/((1/(r_core)) + (1/(x_core)))
z_3 = r_load_p + r_winding + x_winding
z_eq = z_1 + (1/((1/(z_2)) + (1/(z_3))))

# Voltage and currents
v_1 = complex(120, 0)
i_1 = v_1/z_eq
e_1 = v_1 - (z_1*i_1)
i_3 = e_1/z_3
v_2_p = r_load_p*i_3
v_2_s = v_2_p/a
p_in = abs(v_1)*abs(i_1)*np.cos(np.angle(i_1))
p_out = abs(v_2_p)*abs(i_3)*np.cos(np.angle(i_1))
eff = p_out/p_in

problem_idx = find_nearest(r_load_s, 10)

plt.scatter([a*abs(r_load_s)[problem_idx]], [eff[problem_idx]], marker = 'o', color = 'r', zorder = 2)
plt.plot(a*abs(r_load_s), eff, linestyle = '-', zorder = 1)
plt.title('Efficiency of 2:1 Tranformer from 0-200 Ohm Loads')
plt.xlabel('Load Resistance (Seccondary Reference) [Ohm]')
plt.ylabel('Efficiency')
plt.show()

quit()
print('I1: ', end="")
print(i_1)
print('V2 Primary: ', end="")
print(v_2_p)
print('V2 Seccondary: ', end="")
print(v_2_s)

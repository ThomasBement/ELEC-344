# ---------------------------------------- #
# least_squares [Python File]
# Written By: Thomas Bement
# Created On: 2022-12-07
# ---------------------------------------- #

"""
IMPORTS
"""
import numpy as np
"""
CONSTANTS
"""
i_dc = np.array([2.35,3.18,4.66,5.98,7.17,8.27,9.28])
v_dc = 36*np.ones_like([i_dc])
rpm = np.array([2404,2357,2269,2195,2133,2069,2023])
wr = rpm/60*2*np.pi
"""
MAIN
"""
A = np.array([np.transpose(i_dc),np.transpose(wr)]).reshape(7, 2)
b = v_dc.reshape(7, 1)

sol = np.linalg.lstsq(A, b)
R_eq, K_v_eq = sol[0]
print(sol)
print('R_eq',R_eq,'K_v_eq',K_v_eq)      
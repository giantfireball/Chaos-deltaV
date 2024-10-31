# Finding L1, L2, L4 & L5 (CR3BP) using scipy's Newton-Raphson method
# data form JPL-NASA - Park, R.S., et al., 2021
# Aitor Urruticoechea 2022

import numpy as np
from scipy import optimize
from aux_functions import *

try:
    basic_data = np.loadtxt('data/GMdata.txt')
except:
    raise Exception('This code needs data on the G constant and the Planetary masses!')
G = basic_data[0]
GM_sun = basic_data[1]
GM_earth = basic_data[2]
GM_moon = basic_data[3]

mu_earthsun = (GM_earth + GM_moon)/(GM_sun + GM_earth + GM_moon)
mu_earthmoon = (GM_moon)/(GM_earth + GM_moon)



def findL2(guess,mu): #x coordinates (y,z = 0)
    def fun_gamma2(gamma2):
        return 1-mu+gamma2-(1-mu)/((1+gamma2)**2) - mu/(gamma2**2)
    fprime_gamma2 = lambda gamma2: 1- 2*(mu-1)/((1+gamma2)**3) + 2*mu/(gamma2**3)
    root = optimize.newton(fun_gamma2, guess, fprime_gamma2)
    return 1- mu +root

def findL4L5(mu): #x coordinates (y = sqrt(3)/2 NDU, z = 0)
    return 1/2 - mu


L2 = findL2(0.1, mu_earthmoon) #Sun-Moon System
L4 = findL4L5(mu_earthsun) #Sun-Earth System 

np.savetxt('data/LagrangePoints.txt',[L2,L4])

system_plot_moon(mu_earthmoon,L2)
plt.show()

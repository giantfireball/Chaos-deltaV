# Transfers from L2
# Aitor Urruticoechea 2022
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from aux_functions import *
from scipy import linalg

# Basic operational data
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

# Data to be Imported
L2 = 1.155682160772214750
L5 = 0.4999969595765946795

l5 = -mu_earthmoon-(((149597870.700)/(389703000))*(1-mu_earthsun-L5)) ## km vs m why??
l4 = -mu_earthmoon-(((149597870.700)/(389703000))*(1-mu_earthsun-L5))

s_transfer = [1.15433805, -0.0000219492858, -0.0000190481982, 0.00715817243] 

# Initial conditions: periodic orbit
L2_orbits = np.loadtxt('data/L2_LyapunovOrbits.txt')
L4_orbits = np.loadtxt('data/L4_PlanarOrbits.txt')
s0_L4 = np.delete(L4_orbits[-1,:],4,0) # largest
s0 = np.delete(L2_orbits[-1,:],4,0) # largest



#Plots again
#system_plot(mu_earthsun,L2,L5,xlim=[0.4,1.1],ylim=[-1,0.1],zoom=0,showSun=False,showL4=False)
#s_end_orbit, t_orbit = shoot_to_poincare_y(mu_earthsun, s0, style='C0') 

s_obj = [-mu_earthmoon-(((149597870.700)/(389703000))*(1-mu_earthsun-s0_L4[0])), ((149597870.700)/(389703000))*s0_L4[1]]
def objective(ds, s0, s_obj, mu, style='none', step=0.01):
    s_fin, t = shoot_to_poincare_y(mu, [s0[0], s0[1], ds[0], ds[1]], style=style, crossings=2, bstep=step, y_obj=s_obj[1])
    s_f = [s_fin[0], s_fin[1]]
    return np.array(s_f) - np.array(s_obj)

def to_minimize(halfway_point, style_1='none',style_2='none',show=False, step=0.05):
    s_halfway, t_half = shoot_to_poincare_x(mu_earthmoon, s_transfer, style=style_1, crossings=1, x_obj=(halfway_point), bstep=step)
    s0_halfway = [s_halfway[0], s_halfway[1]]
    ds_halfway = [s_halfway[2], s_halfway[3]]
    if show: print('Burn 1 at t: ' + str(t_half))
    success = False
    ds_guess = ds_halfway
    count=0
    while not(success) and count<11: # Retry until solution converges
        count+=1
        root1 = optimize.root(objective, ds_guess, args=(s0_halfway, s_obj, mu_earthmoon,'none',step),method='hybr', tol=10**(-10))
        success = root1.success
        ds_guess = root1.x
    
    #if show: print(s_fin)
    deltav1 = np.array(root1.x)-np.array(ds_halfway)
    if show: print('Frist delta-v (halfway point): ' + str(deltav1) + ' [dx,dy] (absolute: ' + str(np.linalg.norm(deltav1)) + ')')
    
    if show: print('Transfer time: ' + str(t_half))
    
    if not(success): return np.linalg.norm(deltav1) + 100 # If the solution has not converged, the function will return an exagerated response so the solver does not take it as a valid solution
    return np.linalg.norm(deltav1)


plusX = optimize.minimize_scalar(to_minimize, 0.2, bounds=(0.1,0.4),method='bounded',args=('none','none',False,0.02),options={'maxiter':7})

#print('Optimal burn point found at x: ' + str(plusX.x))

Dv1 = to_minimize(plusX.x,style_1='C1',style_2='C2',show=True)
#shoot_to_poincare_x(mu_earthsun, s0_L5, bstep=30, style='black') 
print('For a initial deltaV of: ' + str(Dv1))

legend_object11 = Patch(facecolor='black', edgecolor='black')
legend_object1 = Patch(facecolor='C0', edgecolor='black')
legend_object2 = Patch(facecolor='C1', edgecolor='black')
legend_object3 = Patch(facecolor='C2', edgecolor='black')
plt.legend(handles=[legend_object2, legend_object3, legend_object11],
           labels=['L2-L5 Manifold path', 'Post-correction path', 'Final Planar L5 Orbit'],ncols=1, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,loc='lower right')
plt.show()

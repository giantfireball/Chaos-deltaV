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
L4 = 0.4999969595765946795

l5 = -mu_earthmoon-(((149597870700)/(389703))*(1-mu_earthsun-L4)) ## km vs m why??
l4 = -mu_earthmoon-(((149597870700)/(389703))*(1-mu_earthsun-L4))

# Plots
system_plot_moon(mu_earthmoon,L2,xlim=[0.97,1.02],ylim=[-0.025, 0.025],zoom=1, showEarth=False)
#system_plot(mu_earthsun,L1,L2,L4,L5,xlim=[0.4,1.1],ylim=[-1,0.1],zoom=0,showSun=False,showL4=False)

# Initial conditions: periodic orbit
L2_orbits = np.loadtxt('data/L2_LyapunovOrbits.txt')
L4_orbits = np.loadtxt('data/L4_PlanarOrbits.txt')
s0_L4 = np.delete(L4_orbits[-1,:],4,0) # largest 
s0 = np.delete(L2_orbits[-1,:],4,0) # largest

# Propagation for an orbital period, gives the STM
s_end_orbit, t_orbit = shoot_to_poincare_y(mu_earthmoon, s0, style='C0')
phi = variationals_prop(mu_earthmoon, s0, t_orbit)

# Eigenvalues and eigenvectors of the STM gives the direction of instability
vaps = linalg.eigvals(phi)
#print(vaps) #(this is to check the values are the expected one - or close enough)
veps = linalg.eig(phi)[1]
found = False
n = -1
while not found:
    n = n+1
    if abs(vaps[n])>1.05: found = True
unstable_dir = veps[n,:] / np.linalg.norm(veps[n,:]) # Normalized unstable direction
if unstable_dir[0] > 0: unstable_dir = -unstable_dir
# Exploring that instability
maximum_y = [0,0,0,0]
s_transfer = [0,0,0,0]
h_transfer = 0
for i in range(1,500,5):
    h = i*10**(-7)
    s_new = s0 + h*unstable_dir
    s_end, t_end = shoot_to_poincare_x(mu_earthmoon, s_new, style=None, crossings=1, x_obj=L4)
    shoot(mu_earthmoon, s_new, t_end, style='C1',res=1000)
    if s_end[1] > maximum_y[1]:
        maximum_y = s_end
        s_transfer = s_new
        h_transfer = h

shoot_to_poincare_x(mu_earthmoon, s_transfer, style='C5', bstep=0.1, crossings=1, x_obj=l4) ## L4??## 
print("Manifold's closest approach to L4 at y: " + str(maximum_y[1]))
print("For an initial perturbation of h: " + str(h_transfer))
print("s_transfer: " + str(s_transfer)) 
legend_object1 = Patch(facecolor='C0', edgecolor='black')
legend_object2 = Patch(facecolor='C1', edgecolor='black')
legend_object3 = Patch(facecolor='C5', edgecolor='black')
#plt.legend(handles=[legend_object2, legend_object3],
#           labels=['Trajectories within the L2-L5 Manifold', 'Optimized trajectory'],ncols=1, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,loc='lower right')
s_end_orbit, t_orbit = shoot_to_poincare_y(mu_earthmoon, s0, style='C0')
plt.legend(handles=[legend_object1, legend_object2, legend_object3],
           labels=['Departure L2 Lyapunov Orbit', 'Trajectories within the L2-L5 Manifold', 'Optimized trajectory'],ncols=1, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,loc='lower left')

plt.show()



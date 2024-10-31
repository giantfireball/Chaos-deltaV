# Plotting key Lyapunov orbits around L2
# data form JPL-NASA - Park, R.S., et al., 2021
# Aitor Urruticoechea 2022
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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


# Data to be Imported
L2 = 1.155682160772214750

# JPL gives the initial conditions
x = 1.15386
y = 0
dx = 0
dy = 0.00980825

# Plots
ax = system_plot_moon(mu_earthmoon,L2,xlim=[0.97,1.3],ylim=[-0.025, 0.025],zoom=0, showEarth=False) 
#ax = system_plot(mu_earthsun,L1,L2,L4,L5,xlim=[1.004, 1.016],ylim=[-0.006, 0.006],zoom=2, showSun=False, showL4=False, showL5=False, showEarth = False, showL1=False)

def objective(ds0,s0,mu, style='none', step=0.1):
    s_fin, t = shoot_to_poincare_y(mu, [s0[0],s0[1],ds0[0],ds0[1]], style=style, bstep=step)
    s_f = []
    s_f.append(s_fin[0]+s_fin[2])
    s_f.append(s_fin[0]+s_fin[3])
    return np.array(s_f)-np.array([s0[0]+ds0[0], s0[0]+ds0[1]])


print("") 
print(">> L2 Lyapunov Orbits <<")
print("")

fid1 = open('data/L2_LyapunovOrbits.txt','w')
success_1 = 0
fail_1 = 0
success_2 = 0
fail_2 = 0
total = 0

tol = 10**(-6)
x0 = x

s0 = [x0, y]
ds0 = [dx, dy]
root1 = optimize.root(objective, ds0, args=(s0,mu_earthmoon,'none',0.5))
s_fin, t_1 = shoot_to_poincare_y(mu_earthmoon, [s0[0],s0[1],root1.x[0],root1.x[1]], style='-b')
print('Orbit at: [s0]; ds0; time')
print(s0)
print(str(root1.x[0]) + ' '+ str(root1.x[1]))
print(t_1)
print('')
fid1.write(str(s0[0]) + " " + str(s0[1]) + " " + str(root1.x[0]) + " " + str(root1.x[1]) + " " + str(t_1) + '\n')

count = 1
x0 = x0 + 5*10**(-7)
while count<2001:
    s0 = [x0, y]
    #ds0 = [root1.x[0], root1.x[1]]
    ds0 = [0, root1.x[1]]
    root1 = optimize.root(objective, ds0, args=(s0,mu_earthmoon,'none',0.5), method='hybr', tol=10**(-10))
    s_fin, t_1 = shoot_to_poincare_y(mu_earthmoon, [s0[0],s0[1],root1.x[0],root1.x[1]])
    if root1.success and (np.array([s0[0],s0[1],root1.x[0],root1.x[1]])-np.array(s_fin)<tol).all():
        count = count+1
        x0 = x0 + 2.5*10**(-7)
        if count%200 == 0: #Very small steps are taken, so only 1 in 200 orbits are stored
            print('Orbit at: [s0]; ds0; time')
            print(s0)
            print(str(root1.x[0]) + ' '+ str(root1.x[1]))
            print(t_1)
            shoot(mu_earthmoon,[s0[0],s0[1],root1.x[0],root1.x[1]], t_1, style='-',res=3000)
            fid1.write(str(s0[0]) + " " + str(s0[1]) + " " + str(root1.x[0]) + " " + str(root1.x[1]) + " " + str(t_1) + '\n')
            print('')
    elif not(root1.success and (np.array([s0[0],s0[1],root1.x[0],root1.x[1]])-np.array(s_fin)<tol).all()):
        print('Warning: Solution not good enough (count: ' + str(count)+').' )


fid1.close()
legend_object0 = Patch(facecolor='blue', edgecolor='black')
legend_object1 = Patch(facecolor='C0', edgecolor='black')
legend_object2 = Patch(facecolor='C1', edgecolor='black')
legend_object3 = Patch(facecolor='C2', edgecolor='black')
legend_object4 = Patch(facecolor='C3', edgecolor='black')
legend_object5 = Patch(facecolor='C4', edgecolor='black')
legend_object6 = Patch(facecolor='C5', edgecolor='black')
legend_object7 = Patch(facecolor='C6', edgecolor='black')
legend_object8 = Patch(facecolor='C7', edgecolor='black')
legend_object9 = Patch(facecolor='C8', edgecolor='black')
legend_object10 = Patch(facecolor='C9', edgecolor='black')
plt.legend(handles=[legend_object0, legend_object1, legend_object2, legend_object3, legend_object4, legend_object5, legend_object6,legend_object7, legend_object8, legend_object9, legend_object10],labels=['','','', '','','','','','','', 'Lyapunov Orbits'],ncols=11, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,loc='lower right')
plt.show()

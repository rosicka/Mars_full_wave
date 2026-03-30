# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 16:33:17 2025

@author: student
"""

import numpy as np
import matplotlib.pyplot as plt



latitudes=np.linspace(-90,90,91)
longitudes=np.linspace(0,358,180)
mapa=np.zeros((91,181))
t=np.zeros((91,181))

area=np.zeros((91,180))
for i in range(91):
    for j in range(180):
        area[i,j]=np.sin((2*i)/180*np.pi)*(1/90*np.pi)**2

with open('mars/mapa_20_night.csv', 'r') as data:
    for i in range(90):
        for j in range(181):
            line=data.readline().split(',')
            try:
                mapa[i,j]=float(line[3])
            except:
                mapa[i,j]=np.nan
    line=data.readline().split(',')  
    for j in range(181):
        try:
            mapa[90,j]=float(line[3])
        except:
            mapa[90,j]=np.nan 

         
better_map=np.zeros((91,180))
better_map[:,0:90]=mapa[:,90:180]  
better_map[:,90:180]=mapa[:,0:90]


plt.pcolormesh(longitudes, latitudes, better_map)
plt.colorbar(label='attenuation [dB]')
plt.xlabel('longitude [°]')
plt.ylabel('latitude [°]')
plt.show()

plt.pcolormesh(longitudes, latitudes, np.where(better_map<100,better_map, np.nan))
plt.colorbar(label='attenuation [dB]')
plt.xlabel('longitude [°]')
plt.ylabel('latitude [°]')
plt.show()


limit=100
pod_limit=np.where(better_map<limit,area,0)
print(np.sum(pod_limit)/4/np.pi)
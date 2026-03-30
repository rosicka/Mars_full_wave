# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:28:56 2025

@author: student
"""

import numpy as np
import matplotlib.pyplot as plt
from ionospheric_attenuation_public import cyclotron_freq, plasma_freq_sqr, P_ion, S_ion, D_ion, epsilon_transformed_ion, run_model_ion

e=1.602e-19
m_e=9.109e-31
m_p=1.673e-27
epsilon_0=8.85e-12
c=299792458
    
def electron_density(alts, H, nm, zm):
    ksi=(alts-zm) / H
    return  nm*np.exp(0.5*(1 - ksi - np.exp(-ksi)))

def electron_temperature(alts, T1, T2, z0, H):
    return (T1+T2)/2 + (T2-T1)/2 * np.tanh((alts-z0)/H)

def neutral_density(alts):
    H1 = -100/np.log(0.5e-5)
    ns1 = 2e12 * np.exp(-(alts-100)/H1)
    H2 = -180/ np.log(2e-3) 
    ns2 = 2e9* np.exp(-(alts-120)/H2)
    return ns1+ns2
def electron_neutral_collision_frequency(n,Te):
    return 2.12*10**(-10)*n*np.sqrt(Te)
def ion_neutral_collision_frequency(n,M):
    return 2.6*10**(-9)*n*M**(-1/2)
def electron_ion_collision_frequency(n, Te):
    lmbd=1.23e4*Te**(3/2)*n**(-1/2)
    return 3.62*n*Te**(-3/2)*np.log(lmbd)

plt.rcParams["figure.figsize"] = (3,2)
plt.rcParams['figure.dpi'] = 300
plt.tight_layout()
    
profile_alts=np.linspace(0,350,351)
mag=np.loadtxt('mars/mag_profile.txt')
profile_alts_old=np.flip(np.transpose(mag)[0])
B_0=np.interp(profile_alts,profile_alts_old, np.flip(np.transpose(mag)[1])*10**(-9))
theta_dg=np.interp(profile_alts,profile_alts_old, np.flip(90+np.transpose(mag)[2]))

att=np.zeros(200)
att20=np.zeros(15)
att200=np.zeros(15)
    
#night time
H=30
nm=5000
zm=150

#daytime
SZAs =np.linspace(0,85,18)
#for SZA_dg in SZAs:
# SZA=SZA_dg/180*np.pi
# H=12+3.4*np.log(1/np.cos(SZA))
# nm=160000*np.sqrt(np.cos(SZA))
# zm=125+H*np.log(1/np.cos(SZA))


Hs=np.linspace(20,35,15)
zms=np.linspace(50,215, 34)
nms=np.logspace(2,6,40)
#for nm in nms
#for zm in zms:
#for H in Hs:
electron_density_profile=electron_density(profile_alts, H,nm,zm)


T1 = 100
T2 = 3000
z0 = 200
electron_temperature_profile=electron_temperature(profile_alts, T1,T2, z0, H)

neutral_density_profile=neutral_density(profile_alts)

electron_collision_profile=electron_neutral_collision_frequency(neutral_density_profile, electron_temperature_profile)
ion_collision_profile=ion_neutral_collision_frequency(neutral_density_profile,32)
'''graphs'''
'''    
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax2.plot(theta_dg, profile_alts, label='ϑ')
ax1.plot(B_0*10**9, profile_alts, label='B', color='orange') # Create a dummy plot
ax2.set_xlabel('ϑ [$^\circ$]')
ax1.set_xlabel('B [nT]')
ax1.set_ylabel('altitude [km]')
#ax2.legend()
#ax1.legend()
#ax2.cla()
plt.show()


plt.plot(electron_density_profile, profile_alts)
plt.legend()
plt.xlabel('electron density [cm$^{-3}$]')
plt.ylabel('altitude [km]')
plt.xscale('log')
plt.xlim(10**-6,10000)
#plt.ylim(0,200)
plt.show()

plt.plot(electron_temperature_profile, profile_alts)
plt.xlabel('electron temperature [K]')
plt.ylabel('altitude [km]')
plt.show()

plt.plot(neutral_density_profile, profile_alts)
plt.xlabel('neutral density [cm$^{-3}$]')
plt.ylabel('altitude [km]')
plt.xscale('log')
plt.show()

plt.plot(electron_collision_profile, profile_alts, label='electron neutral')
plt.plot(electron_ion_collision_frequency(electron_density_profile, electron_temperature_profile), profile_alts, label='electron ion')
plt.plot(ion_collision_profile,  profile_alts, label='ion neutral')
plt.plot(m_e/m_p/32*electron_ion_collision_frequency(electron_density_profile, electron_temperature_profile), profile_alts, label='ion electron')
plt.xlabel('collision frequency [Hz]')
plt.ylabel('altitude [km]')
plt.xlim(10**(-6),10**(6))
plt.xscale('log')
plt.show()

plt.plot(cyclotron_freq(B_0, e, m_e)/2/np.pi, profile_alts, label='electron cyclotron f.')
plt.plot(cyclotron_freq(B_0, e, 32*m_p)/2/np.pi, profile_alts, label='ion cyclotron f.')
plt.plot(np.sqrt(plasma_freq_sqr(m_e,e,electron_density_profile*10**6))/2/np.pi, profile_alts, label ='electron plasma f.')
plt.plot(np.sqrt(plasma_freq_sqr(32*m_p,e,electron_density_profile*10**6))/2/np.pi, profile_alts, label = 'ion plasma f.')
plt.xscale('log')
plt.xlim(10**-1, 10**6)
plt.xlabel('frequency [Hz]')
plt.ylabel('altitude [km]')
plt.show()
'''

'Full-wave starts here'
m=[m_e, 32*m_p]
q=[-e,e]

#for polar in ['R', 'L', 'S', 'P']:
polar='P'
#frequencies=np.logspace(1,3,200)
frequencies=np.array([20,200])
#for polar_num in range(4):
for f in frequencies:
    alphas=np.linspace(-75,75,151)
    #for alpha in alphas:
    alpha=0
    kx0=np.sin(alpha/180*np.pi)
    thetas=np.linspace(0,75,76)
    Bs=np.logspace(-6, -3, 35)
    #for thet in thetas:
    #for B_sur in Bs:
    k0=2*np.pi*f/c
      
    
    M=len(profile_alts)
    h=np.zeros_like(profile_alts)
    h[0]=(profile_alts[1]+profile_alts[0])/2*1000
    for i in range(1,M-1):
        h[i]=(profile_alts[i+1]-profile_alts[i-1])/2*1000
    h[M-1]=(profile_alts[M-1]-profile_alts[M-2])/2*1000
    

    S=np.zeros([M], dtype=np.complex128)
    D=np.zeros([M], dtype=np.complex128)
    P=np.zeros([M], dtype=np.complex128)
    epsilon=np.zeros([M,3,3], dtype=np.complex128)

    
    n=np.concatenate((np.reshape(electron_density_profile,(M,1)),np.reshape(electron_density_profile,(M,1))),axis=1)*10**(6)
    ny=np.concatenate((np.reshape(electron_collision_profile, (M,1)),np.reshape(ion_collision_profile, (M,1))), axis=1)
    ei=electron_ion_collision_frequency(electron_density_profile, electron_temperature_profile)
    ei=np.where(np.isnan(ei),0, ei)
    ie=m_e/32/m_p*ei
    # if polar_num==0 or polar_num==1:
    #     ei=np.zeros(M)
    #     ie=np.zeros(M)
    # if polar_num ==0 or polar_num==2:
    #     ny=np.zeros((len(profile_alts),2))
    
    
    B=B_0
    #magn=B_sur/B_0[0]
    #B=B_0*magn
    theta=theta_dg*np.pi/180
    #theta=np.full(M,thet*np.pi/180)
    phi=np.full(M,0)
    
    for i in range(M):
        S[i]=S_ion(B[i], m, q, n[i], ny[i], ei[i], ie[i], f)
        D[i]=D_ion(B[i], m, q, n[i], ny[i], ei[i], ie[i], f)
        P[i]=P_ion(B[i], m, q, n[i], ny[i], ei[i], ie[i], f)
        epsilon[i]=epsilon_transformed_ion(theta[i], phi[i], S[i], D[i], P[i])
    
    alt_init=0
    alt_end=M
    nz_sorted, u, d, F=run_model_ion(kx0, k0, S[alt_init:alt_end], D[alt_init:alt_end], P[alt_init:alt_end], epsilon[alt_init:alt_end], h[alt_init:alt_end], polar)
    
    #attenuation
    amplitude=np.concatenate((u[3],np.array([0,0])))
    fields=np.matmul(F[3], amplitude)
    energy_in=0.5*np.cross(fields[:3], np.conjugate(fields[3:])).real
    
    amplitude=np.concatenate((u[alt_end-alt_init],d[alt_end-alt_init]))
    fields=np.matmul(F[alt_end-alt_init-1], amplitude)
    energy_up=0.5*np.cross(fields[:3], np.conjugate(fields[3:])).real
    

    print(f, -10*np.log10(energy_up[2]/energy_in[2]))
    #i=np.where(Hs==H)
    att[i]=-10*np.log10(energy_up[2]/energy_in[2])
    # if f==20:
    #     att20[i]=-10*np.log10(energy_up[2]/energy_in[2])
    # else:
    #     att200[i]=-10*np.log10(energy_up[2]/energy_in[2])


#         att=np.zeros_like(profile_alts)
#         for i in range(alt_init, alt_end):
#             amplitude=np.concatenate((u[i],np.array([0,0])))
#             fields=np.matmul(F[i-1], amplitude)
#             energy_up=0.5*np.cross(fields[:3], np.conjugate(fields[3:])).real   
#             att[i]=-10*np.log10(energy_up[2]/energy_in[2])
#             if i<90 and att[i]>1:
#                 att[i]=0
#         if polar_num==0:
#             plt.plot(att, profile_alts, label='no collisions')
#         if polar_num==1:
#             plt.plot(att, profile_alts, label='just neutral collisions')
#         if polar_num==2:
#             plt.plot(att, profile_alts, label='just e-i collisions')
#         if polar_num==3:
#             plt.plot(att, profile_alts, label='all collisions')  
# plt.title('200 Hz')
# plt.ylabel('altitude [km]')
# plt.xlabel('attenuation [dB]')
# plt.xlim(0,35)
# plt.show()

#     if polar_num==0:
#         plt.plot(frequencies,att, label='no collisions')
#     if polar_num==1:
#         plt.plot(frequencies,att, label='only neutral collisions')
#     if polar_num==2:
#         plt.plot(frequencies,att, label='only e-i collisions')
#     if polar_num==3:
#         plt.plot(frequencies,att, label='only collisions')
# #plt.ylim(0,100)    
# #plt.legend()
# plt.xscale('log')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Attenuation [dB]')
# plt.show() 

#     if polar=='P':
#         plt.plot(alphas, att, label='p polarisation')
#     else:
#         plt.plot(alphas, att, label='s polarisation')
# plt.title('200 Hz')
# plt.xlabel('α [°]')
# plt.ylabel('attenuation [dB]')
# plt.legend()
# plt.show()

# plt.plot(nms,att20, label='20 Hz')
# plt.plot(nms,att200, label='200 Hz')
# plt.plot(5000,26.01, 'o', color='tab:blue')
# plt.plot(5000,31.08, 'o', color='tab:orange')
# plt.xlabel('peak density [cm$^{-3}$]')
# plt.ylabel('attenuation [dB]')
# plt.xscale('log')
# plt.yscale('log')
# #plt.ylim(10,3000)
# plt.show() 

# plt.plot(zms,att20, label='20 Hz')
# plt.plot(zms,att200, label='200 Hz')
# plt.plot(150,26.01, 'o', color='tab:blue')
# plt.plot(150,31.08, 'o', color='tab:orange')
# plt.xlabel('altitude of maximum density [km]')
# plt.ylabel('attenuation [dB]')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()  
  
# plt.plot(Hs,att20, label='20 Hz')
# plt.plot(Hs,att200, label='200 Hz')
# plt.plot(30,26.01, 'o', color='tab:blue')
# plt.plot(30,31.08, 'o', color='tab:orange')
# plt.xlabel('scale height [km]')
# plt.ylabel('attenuation [dB]')
# plt.xscale('linear')
# plt.yscale('linear')
# plt.show()   

# plt.plot(SZAs,att20, label='20 Hz')
# plt.plot(SZAs,att200, label='200 Hz')
# #plt.title('20 Hz')
# #plt.legend()
# plt.xlabel('SZA [°]')
# plt.ylabel('attenuation [dB]')
# #plt.xscale('log')
# #plt.yscale('log')
# #plt.ylim(10,3000)
# plt.show()

# plt.plot(Bs,att20, label='20 Hz')
# plt.plot(Bs,att200, label='200 Hz')
# plt.plot(B_0[0],26.01, 'o', color='tab:blue')
# plt.plot(B_0[0],31.08, 'o', color='tab:orange')
# #plt.title('20 Hz')
# #plt.legend()
# plt.xlabel('B on the surface [T]')
# plt.ylabel('attenuation [dB]')
# plt.xscale('log')
# #plt.yscale('log')
# #plt.ylim(10,3000)
# plt.show()
    
# plt.plot(thetas,att20, label='20 Hz')
# plt.plot(thetas,att200, label='200 Hz')
# plt.plot(theta_dg[50],26.01, 'o', color='tab:blue')
# plt.plot(theta_dg[50],31.08, 'o', color='tab:orange')
# #plt.title('20 Hz')
# #plt.legend()
# plt.xlabel('ϑ [°]')
# plt.ylabel('attenuation [dB]')
# #plt.xscale('log')
# #plt.yscale('log')
# #plt.ylim(10,3000)
# plt.show()
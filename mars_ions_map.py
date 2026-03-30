# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:28:56 2025

@author: student
"""

import numpy as np
from ionospheric_attenuation_public import P_ion, S_ion, D_ion, epsilon_transformed_ion, run_model_ion

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

profile_alts=np.linspace(0,350,351)

#night time
# H=30
# nm=5000
# zm=150

#daytime
SZA_dg=0
SZA=SZA_dg/180*np.pi
H=12+3.4*np.log(1/np.cos(SZA))
nm=160000*np.sqrt(np.cos(SZA))
zm=125+H*np.log(1/np.cos(SZA))

electron_density_profile=electron_density(profile_alts, H,nm,zm)

T1 = 100
T2 = 3000
z0 = 200
electron_temperature_profile=electron_temperature(profile_alts, T1,T2, z0, H)

neutral_density_profile=neutral_density(profile_alts)

electron_collision_profile=electron_neutral_collision_frequency(neutral_density_profile, electron_temperature_profile)
ion_collision_profile=ion_neutral_collision_frequency(neutral_density_profile,32)

'Full-wave starts here'
m=[m_e, 32*m_p]
q=[-e,e]

polar='P'
f=200
latitudes=np.linspace(-90,90,91) 
longitudes=np.linspace(-180,180,181)
with open('mars/mapa_200_SZA_0.csv', 'a') as output:
    for lat in latitudes:
        for lon in longitudes:
            mag=np.loadtxt('mars/magnetic_profiles_2/lat_'+str(int(lat))+'_lon_'+str(int(lon))+'.csv',delimiter=',')
            profile_alts_old=np.arange(350)
            Br=np.transpose(mag)[0]
            Bthet=np.transpose(mag)[1]
            Bphi=np.transpose(mag)[2]
            Bamp=np.sqrt(Br**2+Bthet**2+Bphi**2)
            theta=np.arccos(Br/Bamp)
            phi=np.arctan(Bphi/Bthet)
            B_0=np.interp(profile_alts,profile_alts_old, Bamp)*10**-9
            theta=np.interp(profile_alts,profile_alts_old, theta)
            phi=np.interp(profile_alts,profile_alts_old, phi)   
            alpha=0
            kx0=np.sin(alpha/180*np.pi)
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
                      
            B=B_0
            phi=np.full(M,0)
            
            for i in range(M):
                S[i]=S_ion(B[i], m, q, n[i], ny[i], ei[i], ie[i], f)
                D[i]=D_ion(B[i], m, q, n[i], ny[i], ei[i], ie[i], f)
                P[i]=P_ion(B[i], m, q, n[i], ny[i], ei[i], ie[i], f)
                epsilon[i]=epsilon_transformed_ion(theta[i], phi[i], S[i], D[i], P[i])
            alt_init=0
            alt_end=M
            nz_sorted, u, d, F=run_model_ion(kx0, k0, S[alt_init:alt_end], D[alt_init:alt_end], P[alt_init:alt_end], epsilon[alt_init:alt_end], h[alt_init:alt_end], polar)

            amplitude=np.concatenate((u[0],np.array([0,0])))
            fields=np.matmul(F[0], amplitude)
            energy_in=0.5*np.cross(fields[:3], np.conjugate(fields[3:])).real

            
            amplitude=np.concatenate((u[alt_end-alt_init],d[alt_end-alt_init]))
            fields=np.matmul(F[alt_end-alt_init-1], amplitude)
            energy_up=0.5*np.cross(fields[:3], np.conjugate(fields[3:])).real
            

            print(f, polar, lat, lon, -10*np.log10(energy_up[2]/energy_in[2]))
            output.write(str(f)+','+str(lat)+','+str(lon)+','+str(-10*np.log10(energy_up[2]/energy_in[2]))+'\n')
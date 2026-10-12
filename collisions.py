# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 13:39:00 2025

@author: student
"""

import numpy as np
import matplotlib.pyplot as plt

e=1.602e-19
m_e=9.109e-31
m_p=1.673e-27
epsilon_0=8.85e-12
c=299792458


def electron_ion_collision_frequency(n, Te):
    lmbd=1.23e4*Te**(3/2)*n**(-1/2)
    return 3.62*n*Te**(-3/2)*np.log(lmbd)
def ion_ion_collision_frequency(n2, M1,M2, Ti, Te, ne, Z1=1,Z2=1):
        lmbd=1.23e4*Te**(1/2)*ne**(-1/2)*Ti
        return 8.50e-2*n2*Z1**2*Z2**2/np.sqrt((M1+M2)*M1/M2)*Ti**(-3/2)*np.log(lmbd)

def create_ei(n,m,Ms,Ti, Te): #v m-3
    M=len(Te)
    N=len(n[0])
    ei=np.zeros((M,N,N))
    for i in range(1,N):
        ei[:,0,i]=electron_ion_collision_frequency(n[:,i]/10**6, Te)
        ei[:,i,0]=m_e/m_p/Ms[i]*ei[:,0,i]
        for j in range(i+1,N):
            ei[:,i,j]=ion_ion_collision_frequency(n[:,j]/10**6, Ms[i], Ms[j] , Ti, Te, n[:,0]/10**6) 
            ei[:,j,i]=n[:,i]*Ms[i]/n[:,j]/Ms[j]*ei[:,i,j]
    ei=np.where(np.isfinite(ei), ei, 0)
    return ei


def electron_neutral_collision_frequency(n,neutrals, Te): #in m-3 #Shung&Nagy
    ny=0
    N=len(neutrals)
    for i in range(N):
        if neutrals[i]=='N2':
            ny+=2.33e-11*(1-1.21e-4*Te)*Te*n[:,i]/10**6
        elif neutrals[i]=='O2':
            ny+=1.81e-10*(1+3.6e-2*np.sqrt(Te))*np.sqrt(Te)*n[:,i]/10**6
        elif neutrals[i]=='O':
            ny+= 8.9e-11*(1+5.7e-4*Te)*np.sqrt(Te)*n[:,i]/10**6           
        elif neutrals[i]=='He':
            ny+=4.6e-10*np.sqrt(Te)*n[:,i]/10**6
        elif neutrals[i]=='H':
            ny+=  4.5e-9*(1-1.35e-4*Te)*np.sqrt(Te)*n[:,i]/10**6       
        elif neutrals[i]=='CO':
            ny+=2.4e-11*(Te+165)*n[:,i]/10**6
        elif neutrals[i]=='CO2':
            ny+=3.68e-8*(1+4.1e-11*abs(4500-Te))**2.93*n[:,i]/10**6
        else:
            ny+=2.12e-10*np.sqrt(Te)*n[:,i]/10**6 #melnik
            print(neutrals[i], 'not implemented, universal value used')
    return ny  

def electron_neutral_collision_frequency_new(n,neutrals, Te): #in m-3 #Cao
    ny=0
    N=len(neutrals)
    for i in range(N):
        if neutrals[i]=='N2':
            qd=3.7e-17*Te**0.46*(1-1.78e-4*Te**0.91)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        elif neutrals[i]=='O2':
            qd=3.75e-19*(1+17.5*Te**0.6-4e-3*Te**1.48)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        elif neutrals[i]=='O':
            qd=9.2e-18*np.exp(1.6*Te**0.09)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]         
        elif neutrals[i]=='He':
            qd=np.exp(-2.5e-5*Te+1.5e-2*Te**0.4-35.26)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        elif neutrals[i]=='H':
            qd=np.exp(2.3e-9*Te**2.2-2e-6*Te**1.5-33.15)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]     
        elif neutrals[i]=='CO': #valid above 770 K
            qd=3.35e-16*(1.78e-11*Te**3-1.98e-7*Te**2+1.2e-3*Te+1.1)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        elif neutrals[i]=='CO2':
            qd=np.exp(-2.3e-11*Te**3+3.5e-7*Te**2-1.87e-3*Te-32)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        elif neutrals[i]=='H2':
            qd=6.8e-16*(1+1.1e-2*Te**0.74-2.1e-3*Te**0.9)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        elif neutrals[i]=='CH4':
            if Te<1350:
                qd=np.exp(1.53e-11*Te**3.6-2.44e-3*Te**1.1-34.2)
            else:
                qd=np.exp(4.2e-11*Te**3-5.6e-7*Te**2+2.77e-3*Te-41)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        elif neutrals[i]=='H2O':
            qd=np.exp(-0.1675*Te**0.58+0.034*Te**0.74-30.6)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        elif neutrals[i]=='Ar':
            if Te<1785:
                qd=np.exp(1.9e-13*Te**3.9-2.7e-3*Te-35.3)
            else:
                qd=np.exp(-1.56e-5*Te**1.5+2e-3*Te-41.6)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        elif neutrals[i]=='N2O':
            if Te<1580:
                qd=np.exp(3.38e-10*Te**3-2.45e-3*Te-33.3)
            else:
                qd=np.exp(-1.68e-7*Te**1.96+5.43e-5*Te**1.33-36.5)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        elif neutrals[i]=='NO2':
            if Te<3300:
                qd=np.exp(-1.47e-3*Te-32.9)+1.84e-15
            else:
                qd=np.exp(-2.7e-11*Te**3+2.32e-7*Te**2-2.16e-4*Te-34.75)
            ny+=0.828334856585605*np.sqrt(Te)*qd*n[:,i]
        else:
            ny+=2.12e-10*np.sqrt(Te)*n[:,i]/10**6 #melnik
            print(neutrals[i], 'not implemented, universal value used')
    return ny             

def nonresonant(M1, M2, alpha):
    if M1>0 and M2>0:
        mu=M1*M2/(M1+M2)
        return 25.879e-16*np.sqrt(alpha/mu)*M2/(M1+M2)
    else:
        return 0
def ion_neutral_collision_frequency(n, ion, neutrals, Te, Ti, Tn):
    ny=np.zeros(len(Ti), dtype=float)
    N=len(neutrals)
    resonances=[]
    if ion=='H+':
        resonances.append('H')
        resonances.append('O')
        Mi=1
    elif ion=='He+':
        resonances.append('He')
        Mi=4
    elif ion=='C+':
        Mi=12
    elif ion=='N+':
        resonances.append('N')
        Mi=14
    elif ion=='O+':
        resonances.append('H')
        resonances.append('O')
        Mi=16
    elif ion=='CO+':
        resonances.append('CO')
        Mi=28
    elif ion=='N2+':
        resonances.append('N2')
        Mi=28
    elif ion=='NO+':
        resonances.append('NO')
        Mi=30
    elif ion=='O2+':
        resonances.append('O2')
        Mi=32
    elif ion=='CO2+':
        resonances.append('CO2')
        Mi=44
    elif ion=='H3+':
        #TODO resonances
        Mi=3
    elif ion=='CH5+':
        #TODO resonances
        Mi=17
    elif ion=='C2H7+':
        #TODO resonances
        Mi=31
    elif ion=='C4H9+':
        #TODO resonances
        Mi=57
    elif ion=='HCO+':
        #TODO resonances
        Mi=29
    else:
        print(ion, 'is not implemented')
        Mi=0
    for neutral in neutrals:
        i=np.where(neutrals==neutral)[0][0]
        if neutral not in resonances:
            if neutral=='H':
                Mn=1
                alpha=0.67
            elif neutral=='He':
                Mn=4
                alpha=0.21
            elif neutral=='N':
                Mn=14
                alpha=1.13
            elif neutral=='O':
                Mn=16
                alpha=0.77
            elif neutral=='CO':
                Mn=28
                alpha=1.97
            elif neutral =='N2':
                Mn=28
                alpha=1.76
            elif neutral=='O2':
                Mn=32
                alpha=1.60
            elif neutral=='CO2':
                Mn=44
                alpha=2.63
            elif neutral=='H2':
                Mn=2
                alpha=0.82
            elif neutral=='CH4':
                Mn=16
                alpha=2.59
            elif neutral=='H2O':
                Mn=18
                alpha=1.48
            elif neutral=='N2O':
                Mn=44
                alpha=3.00
            elif neutral=='Na':
                Mn=23
                alpha=2.70
            elif neutral=='NH3':
                Mn=17
                alpha=2.22
            elif neutral=='NO':
                Mn=30
                alpha=1.74
            elif neutral=='SO2':
                Mn=64
                alpha=3.89
            elif neutral=='Ar':
                Mn=40
                alpha=1.49#https://www.sciencedirect.com/science/article/pii/000926149500005O
            elif neutral=='C2H2':#https://cccbdb.nist.gov/pollistx.asp
                Mn=26
                alpha=3.487
            elif neutral=='C2H4':
                Mn=28
                alpha=4.188
            elif neutral=='C2H6':
                Mn=30
                alpha=4.226
            else:
                print(neutral, 'is not implemented')
            ny+=nonresonant(Mi, Mn, alpha)*n[:,i]
        else:
            Tr=(Ti+Tn)/2
            if neutral=='H':
                if ion=='H+':
                    for j in range(len(Tr)):
                        if Tr[j]>50:
                           ny[j]+=2.65e-10*(1-0.083*np.log10(Tr[j]))**2*np.sqrt(Tr[j])*n[j,i]/10**6
                        else:
                            ny[j]+=nonresonant(1, 1, 0.67)*n[j,i]/10**6
                if ion=='O+':
                    for j in range(len(Tr)):
                        if Tr[j]>300:
                            ny[j]+=4.63e-12*np.sqrt(Tn[j]+Ti[j]/16)*n[j,i]/10**6
                        else:
                            ny[j]+=nonresonant(16, 1, 0.67)*n[j,i]/10**6
            elif neutral=='He':
                for j in range(len(Tr)):
                    if Tr[j]>50:
                        ny[j]+=8.73e-11*(1-0.093*np.log10(Tr[j]))**2*np.sqrt(Tr[j])*n[j,i]/10**6
                    else:
                        ny[j]+=nonresonant(4, 4, 0.21)*n[j,i]/10**6
            elif neutral=='N':
                for j in range(len(Tr)):
                    if Tr[j]>275:
                        ny[j]+=3.83e-11*(1-0.063*np.log10(Tr[j]))**2*np.sqrt(Tr[j])*n[j,i]/10**6
                    else:
                        ny[j]+=nonresonant(14, 14, 1.13)*n[j,i]/10**6
            elif neutral=='O':
                if ion=='O+':
                    for j in range(len(Tr)):
                        if Tr[j]>235:
                            ny[j]+=3.67e-11*(1-0.064*np.log10(Tr[j]))**2*np.sqrt(Tr[j])*n[j,i]/10**6
                        else:
                            ny[j]+=nonresonant(16, 16, 0.77)*n[j,i]/10**6
                if ion=='H+':
                    for j in range(len(Tr)):
                        if Tr[j]>300:
                            ny[j]+=6.61e-11*(1-0.047*np.log10(Ti[j]))**2*np.sqrt(Ti[j])*n[j,i]/10**6
                        else:
                            ny[j]+=nonresonant(1, 16, 0.77)*n[j,i]/10**6
            elif neutral=='N2':
                for j in range(len(Tr)):
                    if Tr[j]>170:
                        ny[j]+=5.14e-11*(1-0.069*np.log10(Tr[j]))**2*np.sqrt(Tr[j])*n[j,i]/10**6
                    else:
                        ny[j]+=nonresonant(28, 28, 1.76)*n[j,i]/10**6
            elif neutral=='O2':
                for j in range(len(Tr)):
                    if Tr[j]>800:
                        ny[j]+=2.59e-11*(1-0.073*np.log10(Tr[j]))**2*np.sqrt(Tr[j])*n[j,i]/10**6
                    else:
                        ny[j]+=nonresonant(32, 32, 1.60)*n[j,i]/10**6
            elif neutral=='CO':
                for j in range(len(Tr)):
                    if Tr[j]>525:
                        ny[j]+=3.42e-11-(1-0.085*np.log10(Tr[j]))**2*np.sqrt(Tr[j])*n[j,i]/10**6
                    else:
                        ny[j]+=nonresonant(28, 28, 1.97)*n[j,i]/10**6
            elif neutral=='CO2':
                for j in range(len(Tr)):
                    if Tr[j]>850:
                        ny[j]+=2.85e-11*(1-0.083*np.log10(Tr[j]))**2*np.sqrt(Tr[j])*n[j,i]/10**6
                    else:
                        ny[j]+=nonresonant(44, 44, 2.63)*n[j,i]/10**6
            else:
               print(neutral, ion, 'resonance is not implemented') 
    return ny
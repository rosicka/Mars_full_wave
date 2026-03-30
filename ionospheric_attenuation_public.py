# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.optimize import fsolve


e=1.602e-19
m_e=9.109e-31
m_p=1.673e-27
epsilon_0=8.85e-12
c=299792458

def plasma_freq_sqr(m,q,n):
    return q**2*n/epsilon_0/m
def cyclotron_freq(B,q,m):
    return q*B/m


'''m, q, n, ny are arrays with each element representing one particle specie'''

'''Stix parameters including electron-ions collisions'''
#using the convention of wave field changing as exp(-i omega t+i kr) according to Letheinen et al. 2008, 2009
def P_ion(B, m, q, n, ny, ei, ie, f):
    omega=2*np.pi*f
    lambdae=-complex(0,1)*omega+ny[0]+ei
    lambdai=-complex(0,1)*omega+ny[1]+ie
    sigma=(plasma_freq_sqr(m[0],q[0],n[0])*(lambdai-ie)+plasma_freq_sqr(m[1],q[1],n[1])*(lambdae-ei))/(lambdae*lambdai-ei*ie)
    return 1+complex(0,1)/omega*sigma

def S_ion(B, m, q, n, ny, ei, ie, f):
    omega=2*np.pi*f
    lambdae=-complex(0,1)*omega+ny[0]+ei
    lambdai=-complex(0,1)*omega+ny[1]+ie
    omegae=cyclotron_freq(B,q[0],m[0])
    omegai=cyclotron_freq(B,q[1],m[1])
    D=lambdae**2*lambdai**2+lambdae**2*omegai**2+omegae**2*lambdai**2+omegae**2*omegai**2+ei**2*ie**2+2*omegae*omegai*ei*ie-2*lambdae*lambdai*ei*ie
    sigmap=n[0]*e/B/D*(lambdae*lambdai-omegae*omegai-ei*ie)*(omegai*(lambdae-ei)-omegae*(lambdai-ie))
    return 1+complex(0,1)/omega/epsilon_0*sigmap
    
def D_ion(B, m, q, n, ny, ei, ie, f):
    omega=2*np.pi*f
    lambdae=-complex(0,1)*omega+ny[0]+ei
    lambdai=-complex(0,1)*omega+ny[1]+ie
    omegae=cyclotron_freq(B,q[0],m[0])
    omegai=cyclotron_freq(B,q[1],m[1])
    D=lambdae**2*lambdai**2+lambdae**2*omegai**2+omegae**2*lambdai**2+omegae**2*omegai**2+ei**2*ie**2+2*omegae*omegai*ei*ie-2*lambdae*lambdai*ei*ie
    sigmah=n[0]*e/B/D*(omegae**2*lambdai**2-omegai**2*lambdae**2-(omegae*ie-omegai*ei)*(omegae*lambdai+omegai*lambdae))
    return 1/omega/epsilon_0*sigmah

'''transformation of plasma permitivity into general direction of insintic B_0'''
def epsilon_transformed_ion(theta, phi, S, D, P):
    T=np.array([[np.cos(theta)*np.cos(phi), -np.sin(phi), np.sin(theta)*np.cos(phi)], [np.cos(theta)*np.sin(phi),np.cos(phi),np.sin(theta)*np.sin(phi)],[-np.sin(theta), 0, np.cos(theta)]])
    epsilon=np.array([[S, -complex(0,1)*D, 0], [complex(0,1)*D, S, 0], [0,0,P]])
    return np.matmul(np.matmul(T,epsilon),np.linalg.inv(T))

     

'''computation of nz'''
def count_nz_ion(kx0,S,D_stix,P,epsilon):
    A=-epsilon[2,2]
    B=-kx0*(epsilon[2,0]+epsilon[0,2])
    C=-kx0**2*(epsilon[2,2]+epsilon[0,0])+epsilon[1,1]*epsilon[2,2]+epsilon[0,0]*epsilon[2,2]-epsilon[0,2]*epsilon[2,0]-epsilon[1,2]*epsilon[2,1]
    D=kx0*(epsilon[0,2]*epsilon[1,1]+epsilon[2,0]*epsilon[1,1]-epsilon[0,1]*epsilon[1,2]-epsilon[1,0]*epsilon[2,1])-kx0**3*(epsilon[0,2]+epsilon[2,0])
    E=-(S**2-D_stix**2)*P-kx0**4*epsilon[0,0]+kx0**2*(epsilon[0,0]*epsilon[2,2]+epsilon[0,0]*epsilon[1,1]-epsilon[0,1]*epsilon[1,0]-epsilon[0,2]*epsilon[2,0])
    return(np.roots([A,B,C,D,E]))

'''sorting of complex nz according to increasing imaginary part'''
def sort_nz(roots):
    nz_sorted=np.sort_complex(roots)[::-1]
    if nz_sorted[0].imag<nz_sorted[1].imag:
        mem=nz_sorted[1]
        nz_sorted[1]=nz_sorted[0]
        nz_sorted[0]=mem
    if nz_sorted[2].imag<nz_sorted[3].imag:
        mem=nz_sorted[2]
        nz_sorted[2]=nz_sorted[3]
        nz_sorted[3]=mem            
    return nz_sorted

'''computation of matrices F, P according to Letheinen et al. 2008, 2009'''
def count_F(kx0, nz, epsilon):
    F=np.zeros([6,4], dtype=np.complex128)
    for j in range(4):
        a=np.array([[nz[j]**2,0,-nz[j]*kx0],[0,kx0**2+nz[j]**2,0],[-nz[j]*kx0,0,kx0**2]])-epsilon
        values, vectors=np.linalg.eig(a)
        vectors=np.transpose(vectors)
        for i in range(len(values)):
            if i==np.argmin(abs(values)):
                if vectors[i,0].real>(vectors[i,0]*complex(0,1)).real:
                    F[0,j]=vectors[i,0]
                    F[1,j]=vectors[i,1]
                    F[2,j]=vectors[i,2]
                    F[3,j]=-nz[j]*vectors[i,1]
                    F[4,j]=nz[j]*vectors[i,0]-kx0*vectors[i,2]
                    F[5,j]=kx0*vectors[i,1]
                else:
                    F[0,j]=vectors[i,0]*complex(0,1)
                    F[1,j]=vectors[i,1]*complex(0,1)
                    F[2,j]=vectors[i,2]*complex(0,1)
                    F[3,j]=-nz[j]*vectors[i,1]*complex(0,1)
                    F[4,j]=nz[j]*vectors[i,0]*complex(0,1)-kx0*vectors[i,2]*complex(0,1)
                    F[5,j]=kx0*vectors[i,1]*complex(0,1)
    return(F)
def F_square(F):
    return np.array([F[0], F[1], F[3], F[4]])
def count_Pu(k0,h,nz):
    return np.array([[np.exp(complex(0,1)*k0*nz[0]*h),0],[0,np.exp(complex(0,1)*k0*nz[1]*h)]])
def count_Pd(k0,h,nz):
    return np.array([[np.exp(-complex(0,1)*k0*nz[2]*h),0],[0,np.exp(-complex(0,1)*k0*nz[3]*h)]])

'''completion of the model'''
def run_model_ion(kx0,k0, S,D, P, epsilon, h, polar):
    M=len(S)
    nz_sorted=np.zeros([M,4], dtype=np.complex128)
    F=np.zeros([M,6,4], dtype=np.complex128)
    F_sqr=np.zeros([M,4,4], dtype=np.complex128)
    Pu=np.zeros([M,2,2], dtype=np.complex128)
    Pd=np.zeros([M,2,2], dtype=np.complex128)
    Tu=np.zeros([M-1,4,4], dtype=np.complex128)
    Td=np.zeros([M-1,4,4], dtype=np.complex128)
    Ru=np.zeros([M+1,2,2], dtype=np.complex128)
    U=np.zeros([M,2,2], dtype=np.complex128)
    u=np.zeros([M+1,2], dtype=np.complex128)
    d=np.zeros([M+1,2], dtype=np.complex128)
    #inicialization of incident wave polarisation
    if polar=='R':
        u[0]=np.array([0,1])
    elif polar=='L': 
        u[0]=np.array([1,0])
    elif polar=='P':
        u[0]=np.array([1,1])
    elif polar=='S':
        u[0]=np.array([1,-1])
    for k in range(M):
        nz_sorted[k]=sort_nz(count_nz_ion(kx0,S[k],D[k],P[k],epsilon[k]))
        F[k]=count_F(kx0, nz_sorted[k], epsilon[k])
        #polarisation check
        if (F[k,1,0]/F[k,0,0]).imag >0:
            mem=nz_sorted[k,0]
            nz_sorted[k,0]=nz_sorted[k,1]
            nz_sorted[k,1]=mem
            for j in range(6):
                mem=F[k,j,0]
                F[k,j,0]=F[k,j,1]
                F[k,j,1]=mem
        if (F[k,1,3]/F[k,0,3]).imag >0:
            mem=nz_sorted[k,2]
            nz_sorted[k,2]=nz_sorted[k,3]
            nz_sorted[k,3]=mem
            for j in range(6):
                mem=F[k,j,2]
                F[k,j,2]=F[k,j,3]
                F[k,j,3]=mem            
        F_sqr[k]=F_square(F[k])
        Pu[k]=count_Pu(k0,h[k],nz_sorted[k])
        Pd[k]=count_Pd(k0,h[k],nz_sorted[k])
    for k in range(M-1):
        Tu[k]=np.matmul(np.linalg.inv(F_sqr[k+1]),F_sqr[k])
        Td[k]=np.matmul(np.linalg.inv(F_sqr[k]),F_sqr[k+1])
    for k in range(M-2,-1,-1):
        prvniu=Td[k,2:,:2]+np.matmul(Td[k,2:,2:],Ru[k+1])
        druhyu=Td[k,:2,:2]+np.matmul(Td[k,:2,2:],Ru[k+1])
        Ru[k]=np.matmul(np.matmul(Pd[k],prvniu),np.matmul(np.linalg.inv(druhyu),Pu[k]))
        U[k]=np.matmul(Tu[k,:2,:2], Pu[k])+np.matmul(np.matmul(Tu[k,:2,2:],prvniu),np.matmul(np.linalg.inv(druhyu),Pu[k]))
    for k in range(1,M):
        u[k]=np.matmul(U[k-1],u[k-1])
        for k in range(M):
            d[k]=np.matmul(Ru[k], u[k])
    u[M]=np.matmul(Pu[M-1],u[M-1])
    return nz_sorted, u, d, F


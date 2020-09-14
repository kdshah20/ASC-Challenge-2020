# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:57:59 2020

@author: karan
"""
import numpy as np
def get_Ex(orient):
    
    Q=[]
    S=[]
    #orient=[45,90,-45,0]
    for i in orient:
        E1l=170000
        E2l=9000
        E3l=9000
        v12l=0.34
        v21l=(v12l*E2l)/E1l
        v13l=0.34
        v31l=(v13l*E1l)/E3l
        v23l=0.34
        v32l=(v23l*E3l)/E2l
        G12l=4800
        G23l=4500
        G31l=4800
        theta=i*np.pi/180
        c=np.cos(theta)
        s=np.sin(theta)
        Ql = 1/(1-v12l*v21l)*np.array([[E1l,v21l*E1l,0],
                                       [v12l*E2l,E2l,0],
                                       [0,0,G12l*(1-v12l*v21l)]])
        T1=np.array([[c**2,s**2,-2*s*c], 
                     [s**2,c**2,2*s*c],
                     [s*c,-s*c,c**2-s**2]])
        T=np.array([[c**2,s**2,0,0,0,s*c], 
                    [s**2,c**2,0,0,0,-s*c],
                    [0,0,1,0,0,0],
                    [0,0,0,c,s,0],
                    [0,0,0,-s,c,0],
                    [-2*c*s,2*c*s,0,0,0,c**2-s**2]])
        Tsig=np.array([[c**2,s**2,0,0,0,2*s*c], 
                       [s**2,c**2,0,0,0,-2*s*c],
                       [0,0,1,0,0,0],
                       [0,0,0,c,s,0],
                       [0,0,0,-s,c,0],
                       [-c*s,c*s,0,0,0,c**2-s**2]])
        s11=1/E1l
        s12=-v21l/E2l
        s13=-v31l/E3l
        s21=-v12l/E1l
        s22=1/E2l
        s23=-v32l/E3l
        s31=-v13l/E1l
        s32=-v23l/E2l
        s33=1/E3l
        s44=1/G31l
        s55=1/G23l
        s66=1/G12l
        S1=np.array([[s11,s12,s13,0,0,0],
                     [s21,s22,s23,0,0,0],
                    [s31,s32,s33,0,0,0],
                    [0,0,0,s44,0,0],
                    [0,0,0,0,s55,0],
                    [0,0,0,0,0,s66]])
        S.append(np.dot(np.dot(np.linalg.inv(T),S1),(Tsig)))
        Q.append(np.dot(np.dot(T1,Ql),np.transpose(T1)))
    #print(Q)
    props=np.zeros(1)
    for ii in range(0,len(orient)):
        Sstar = S[ii]
        Ex = 1/Sstar[0,0]
        props[0]+=Ex
    Ex_effective=props[0]/len(orient)
    #print('\nEffective properties for '+str(orient))
    #print('Ex '+str(props[0]/len(orient)))
    return Ex_effective

thetaval=[00, 15, 30, 45, 60, 75, 90]
thetaval2=[15, 30, 60, 75, 00, 45, 90]
stackingseq=[]
Ex_eff=[]
for j in range(1,488):
    theta1=thetaval[(j-1)%7]
    theta2=thetaval[int((j-1)/7)%7]    
    theta3=thetaval2[int((j-1)/(7*7))%7]
    theta4=thetaval2[int((j-1)/(7*7*7))%7]
    theta=[theta1,theta2,theta3,theta4,theta4,theta3,theta2,theta1]
    Ex_eff.append(get_Ex(theta))
    stackingseq.append(theta)
   
thetaval=[00, 15, 30, 45, 60, 75, 90]
thetaval2=[15, 60, 45, 30, 75, 00, 90]
thetaval3=[60, 75, 00, 45, 90, 15, 30]
thetaval4=[15, 30, 60, 75, 00, 45, 90]

for j in range(1,638):    
    theta4=thetaval4[(j-1)%7]
    theta3=thetaval3[int((j-1)/7)%7]    
    theta1=thetaval[int((j-1)/(7*7))%7]
    theta2=thetaval2[int((j-1)/(7*7*7))%7]
    theta=[theta1,theta2,theta3,theta4,theta4,theta3,theta2,theta1]
    Ex_eff.append(get_Ex(theta))
    stackingseq.append(theta)
for j in range(1,8): 
    theta=[thetaval[j-1],thetaval[j-1],thetaval[j-1],thetaval[j-1],thetaval[j-1],thetaval[j-1],thetaval[j-1],thetaval[j-1]]
    #print(theta)
    Ex_eff.append(get_Ex(theta))
    stackingseq.append(theta)
    
np.savetxt('C:/Users/shahk/Documents/ASC Sim 2019/Input data/Effective_stiffness.csv',Ex_eff,delimiter=',')
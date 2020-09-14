# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:45:59 2020

@author: shahk
"""

import random
import numpy as np
import math
import codecs
import re
import pylab as p


# Getting data for different laminate sequences
Lam_stackseq= []

infile = open("C:/Users/shahk/Documents/ASC Sim 2019/Input data/theta_values.txt", 'r')
for line in infile:
    ls = re.sub('\s+', ' ', line)
    #a = np.array(ast.literal_eval(ls))
    #type(a)
    #stack_seq.append(a)
    k = np.fromstring(ls[1:-1],dtype=np.float,sep=',')
    Lam_stackseq.append(k)
    
Effec_stiff = np.loadtxt("C:/Users/shahk/Documents/ASC Sim 2019/Input data/Effective_stiffness.csv",delimiter=',')

data = np.concatenate((Lam_stackseq, Effec_stiff.reshape(len(Effec_stiff),1), np.zeros((len(Lam_stackseq),1))), axis = 1)
    

# Getting strength data for laminates

Lam_strength = []
file = open("C:/Users/shahk/Documents/ASC Sim 2019/Input data/strength_values.txt", 'r')
for line in file:
    ls1 = re.sub('\s+', ' ', line)
    k1 = float(ls1)
    Lam_strength.append(k1)

Lam_strength = np.asarray(Lam_strength)
Lam_strength = Lam_strength.reshape(len(Lam_strength),1)
Lam_stackseq = np.asarray(Lam_stackseq)

lam_data = np.concatenate((data,Lam_strength),axis=1)

np.savetxt("C:/Users/shahk/Documents/ASC Sim 2019/Input data/laminate_data.csv",lam_data, delimiter=',')

# Getting data for open hole laminate sequences
ohT_seq = []

infile = open("C:/Users/shahk/Documents/ASC Sim 2019/Input data/theta_values-ohT.txt", 'r')
for line in infile:
    ls = re.sub('\s+', ' ', line)
    #a = np.array(ast.literal_eval(ls))
    #type(a)
    #stack_seq.append(a)
    k = np.fromstring(ls[1:-1],dtype=np.float,sep=',')
    ohT_seq.append(k)    

ohT_seq = np.asarray(ohT_seq)

hole_present = np.ones((len(ohT_seq),1))

Effec_stiff_ohT = np.loadtxt("C:/Users/shahk/Documents/ASC Sim 2019/Input data/effective-Ex-ohT.txt",delimiter=',')

ohT_data = np.concatenate((ohT_seq,Effec_stiff_ohT.reshape(len(Effec_stiff_ohT),1),hole_present),axis=1)
ohT_data = ohT_data[0:183]

# Getting strength data for open hole laminates

ohLam_strength = []
file2 = open("C:/Users/shahk/Documents/ASC Sim 2019/Input data/strength_values-ohT.txt", 'r')
for line in file2:
    ls1 = re.sub('\s+', ' ', line)
    k1 = float(ls1)
    ohLam_strength.append(k1)

ohLam_strength = np.asarray(ohLam_strength)
ohLam_strength = ohLam_strength.reshape(len(ohLam_strength),1)

ohT_data = np.concatenate((ohT_data,ohLam_strength),axis=1)

np.savetxt("C:/Users/shahk/Documents/ASC Sim 2019/Input data/laminate_data_ohT.csv",ohT_data, delimiter=',')

total_data = np.concatenate((lam_data,ohT_data),axis =0)

np.savetxt("C:/Users/shahk/Documents/ASC Sim 2019/Input data/totaldata.txt",total_data,delimiter =',')

np.savetxt("C:/Users/shahk/Documents/ASC Sim 2019/Input data/MLdata_ang.csv",data,delimiter=',')

X = 2050e6;
Y=190e6;
S=81e6;
#k_array = []
Lay_strg = np.empty((len(Lam_stackseq),8,1)); 
for j in range(len(Lam_stackseq)):
    the = Lam_stackseq[j];
    Sx_array = np.zeros((len(the),1));
    l = np.where(the == 0);
    #k_array.append(l[0][0]);
    for i in range(len(the)):
        temp=the[i];
        m_temp = math.cos(temp*math.pi/180);
        n_temp = math.sin(temp*math.pi/180);
        denom = m_temp**4/(X**2) + n_temp**4/(Y**2)+ (m_temp**2)*(n_temp**2)*((1/S**2) - (1/X**2)); 
        Sx_array[i] = math.sqrt(1/denom)
    Lay_strg[j]= Sx_array/1e6;

Lay_strg2 = np.transpose(np.hstack(Lay_strg));

data2 = np.concatenate((Lay_strg2,Effec_stiff.reshape(len(Effec_stiff),1),Lam_strength), axis=1);
np.savetxt('C:/Users/shahk/Documents/ASC Sim 2019/MLdata_laystrgh.csv',data2,delimiter=',')

ang = np.arange(91);
lamina_strg = np.zeros((len(ang),1))
X = 2050e6;
Y=190e6;
S = 81e6;

for i in range(len(ang)):
    m = math.cos(i*math.pi/180);
    n = math.sin(i*math.pi/180);
    den = m**4/(X**2) + n**4/(Y**2)+(m**2)*(n**2)*((1/S**2) - (1/X**2)); 
    lamina_strg[i] = math.sqrt(1/den)    
laminastrg = np.concatenate((ang.reshape(len(ang),1),lamina_strg), axis=1)    
np.savetxt('C:/Users/shahk/Documents/ASC Sim 2019/lamina_strength_new.csv', laminastrg, delimiter = ',')

p.figure()
p.plot(ang,lamina_strg/1e6)
p.xlabel(r'Fiber angle ($\theta /deg$)')
p.xlim(0,90)
p.ylim(0,2100)
p.ylabel(r'Stress, $\sigma_x$ (MPa)')
p.show()
p.savefig('Tsai-hill.png')
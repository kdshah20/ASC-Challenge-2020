# Program to find failure prediction using Hashin failure in an open hole orthotropic composite plate.
# For details of the analytical solution Refer - Appl Mech 88, 1187�1208 (2018). https://doi.org/10.1007/s00419-018-1366-x

__desc__='''This program predcits failure in an open hole orthotropic plate lamina using Hashin Failure
Input Elastic modulii E1,E2, G12, v12, nominal stresses S1,S2,S12, fiber angle (theta_in_deg) with respect to vertical axis, Strength of the lamina Xt,Xc,Yt,Yc,Sl,St
in MPa units
where Xt is fiber strengt in tension
Xc is fiber strength in compression
Yt is matrix strength in tension
Yc is matric strength in compression
Sl is longitudinal shear strength 
St is transverse shear strength
refer Hashin,  Z., “Failure Criteria for Unidirectional Fiber Composites,” Journal of Applied Mechanics, vol. 47, pp. 329–334, 1980.
and Abaqus documention for Hashin failure
'''

import math
import numpy as np
import argparse

#For finding stresses in a single ply, input modulus in the fiber direction (E1), transverse modulus E2
# inplane shear modulus (G12) inplane poissons ratio (v12), nominal stresses at the boundary and fiber angle with respect to vertical.
# For a multilayered symmetric balanced laminate composite plate, replace E1 by (A11*A22-A12**2)/(t*A22)
#E2 by (A11*A22-A12**2)/(t*A11), G12 by A33/t, v21 by A21/A11 and theta = 0; where Aij are the components of the ABD matrix
#E1/G12>2*v12
def hoop_stress(E1,E2, G12, v12, S1,S2,S12,theta_in_deg):
    S = {}
    alfa = np.linspace(0, math.pi, 1000)
    #S1 is the nominal stress in the vertical direction
    #S2 is the nominal stress in the horizontal direction
    #S12 is the shear stress
    #theta_in_deg is the angle with respect to vertical axis in degrees
    theta=theta_in_deg*math.pi/180
    v21=(E2/E1)*v12
    eta=E2/(2*G12)-v21
    mu =E2/E1
    if eta**2<mu:
        print("Elastic moduli of the orthotropic composite doesn't satisfy the condition\
         (E2/2G12-v21)^2 > E2/E1")
        raise ValueError
    #defining constants/
    #-------------------------------------
    nu =np.sqrt(eta-np.sqrt(eta**2-mu))
    psi=np.sqrt(eta+np.sqrt(eta**2-mu))
    gamma1=(psi-1)/(psi+1)
    gamma2=(nu-1)/(nu+1)
    for a in alfa:
        omega = (1 + gamma1 ** 2 - 2 * gamma1 * np.cos(2 * (a - theta))) * (1 + gamma2 ** 2 - 2 * gamma2 * np.cos(2 * (a - theta)))
#Biaxial loading case
        N1=(1+gamma1)*(1+gamma2)*(1+gamma1+gamma2-gamma1*gamma2-2*np.cos(2*(a-theta)))
        N2=(1-gamma1)*(1-gamma2)*(1-gamma1-gamma2-gamma1*gamma2+2*np.cos(2*(a-theta)))
        N3=4*(gamma1*gamma2-1)*np.sin(2*(a-theta))
        M1=(S1*(1+np.cos(2*theta))+S2*(1-np.cos(2*theta)))/2
        M2=(S1*(1-np.cos(2*theta))+S2*(1+np.cos(2*theta)))/2
        M3=((S1-S2)*np.sin(2*theta))/2

   #Pure shear loading case
        n1=(1+gamma1)*(1+gamma2)*(1+gamma1+gamma2-gamma1*gamma2-2*np.cos(2*(a-theta)))*np.sin(2*theta)
        n2=(1-gamma1)*(1-gamma2)*(1-gamma1-gamma2-gamma1*gamma2+2*np.cos(2*(a-theta)))*np.sin(2*theta)
        n3=4*(gamma1*gamma2-1)*np.sin(2*(a-theta))*np.cos(2*theta)

    #Total stress
        sigma = ((M1 * N1 + M2 * N2 - M3 * N3) + (n1 - n2 + n3) * S12) / omega # circumferential stress aroung the circular hole for 0 deg to 180 degree from the vertical
        sigma_1 = ((np.sin(a - theta))**2)*sigma # stress in the fiber direction
        sigma_2 = ((np.cos(a - theta))**2)*sigma # stress in the matrix direction
        sigma_12= -(np.cos(a-theta))*(np.sin(a-theta))*sigma # inplane shear stress 
        S.update({a: (sigma_1,sigma_2,sigma_12)})
    return (S) 
#-------------------------------------------------------------------------------------------
# Hashin failure criterion
def Hashin(S,Xt,Xc,Yt,Yc,Sl,St):
    f = {}
    for angle, stress in S.items():
        # Fiber failure
        if stress[0] >= 0:
             Ft = (stress[0] / Xt)**2 # Fiber tension
             Fiber_failure_factor = (Ft, 1)
        else:
             Fc = (stress[0] / Xc)**2     # Fiber compression
             Fiber_failure_factor = (Fc, -1)
             
        # Matrix failure
        if stress[1] >= 0:
             Mt = (stress[1] / Yt)**2 + (stress[2] / Sl)**2  #  Matrix tension
             Matrix_failure_factor = (Mt, 2)
        else:
             Mc = (stress[1]/ (2*St))**2 + (stress[1] / Yc) * ((Yc/(2*St))**2 -1) + (stress[2]/Sl)**2 # Matrix compression
             Matrix_failure_factor = (Mc, -2)
      
       # Maximum failure index
        if Fiber_failure_factor[0]>Matrix_failure_factor[0]:
             Failure_factor = Fiber_failure_factor
        else:
             Failure_factor = Matrix_failure_factor
        f.update({Failure_factor[0]:(angle,Failure_factor[1])})

    f_max = max(f)
    return (f_max,f[f_max])



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__desc__)
    parser.add_argument('-E1', default = 170000, type=float, help='Elastic moduli E1')
    parser.add_argument('-E2', default = 9000 , type=float,help='Elastic moduli E2')
    parser.add_argument('-G12',default = 4800 , type=float, help='Elastic moduli G12')
    parser.add_argument('-v12', default = 0.34, type=float,help='Elastic moduli v12')
    parser.add_argument('-Xt', default = 2050, type=float,help='Fiber tensile strength')
    parser.add_argument('-Xc', default=1200, type=float, help = 'Fiber comprssive strength')
    parser.add_argument('-Yt', default=62,type=float, help = 'Matrix tensile strength')
    parser.add_argument('-Yc', default=190,type=float, help = 'Matrix compressive strength')
    parser.add_argument('-Sl', default=81,type=float, help = 'Longitudinal shear strength')
    parser.add_argument('-St', default=81,type=float, help = 'Transverse shear strength')
    parser.add_argument('-S1', type=float, help = 'Far field stress in X direction (Nx/width of the lamina)', required=True)
    parser.add_argument('-S2', default=0, type=float, help = 'Far field stress in Y direction (Ny/Length of the lamina)')
    parser.add_argument('-S12', default=0, type=float, help = 'Far field shear stress  (Nxy/width of the lamina)')
    parser.add_argument('-theta', type =float, help = 'Fiber angle in degrees', required = True)
    arg = parser.parse_args()
    E1,E2,G12,v12,Xt,Xc,Yt,Yc,Sl,St,S1,S2,S12,theta = arg.E1,arg.E2,arg.G12,arg.v12,arg.Xt,arg.Xc,arg.Yt,arg.Yc,arg.Sl,arg.St,arg.S1,arg.S2,arg.S12,arg.theta
    S = hoop_stress(E1,E2, G12, v12, S1,S2,S12,theta)
    H = Hashin(S,Xt,Xc,Yt,Yc,Sl,St)
    if H[1][1] == 1:
        index = "fiber failure in tension"
    elif H[1][1] == -1:
        index = "fiber failure in compression"
    elif H[1][1] == 2:
        index = "Matrix failure in tension"
    elif H[1][1] == -2:
        index = "Matrix failure in compression"

    angle = H[1][0]*180/(math.pi)

    if H[0] >1:
        print (" Failure initiated in the lamina at an angle of "+str(angle)+" degrees from the vertical"+ "\n The failure mode is " +index+'\n The failure index is ' + str(H[0]))
    elif H[0] <= 1:
        print ("Maximum strength has not reached")
        

# While
     #   True:
      #      if f_max > 0.98:
      #          S1 = S1-S1 / 2
       #         f_max = Hashin(Xt,Xc,Yt,Yc,Sl,St)
      #      else if fmax < 0.98:
        #        S1 = S1+S1 / 2
        #        f_max = Hashin(Xt, Xc, Yt, Yc, Sl, St)
        #    else:
          #      break:


    #------------------------------------










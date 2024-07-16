#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:17:51 2024

@author: corneliasheeran
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:23:51 2024

@author: corneliasheeran
"""

import sympy as sym
import numpy as np
from sympy import MatrixSymbol, Inverse
import sys
import math
from sympy.matrices import Matrix
from sympy import symbols, latex
from sympy import Matrix, matrix_multiply_elementwise
from sympy import *
import seaborn as sns
import matplotlib.pyplot as plt

def sp_partial_inversion(m, *cells):
    ''' Partial inversion algorithm.
    ARGUMENTS
        m <sympy.Matrix> : symbolic matrix
        *cells <tuple>   : 2-tuples with matrix indices to perform partial inversion on.
    RETURNS
        <sympy.Matrix>   : matrix m partially-inverted at indices *cells
    '''
    # Partial inversion algorithm
    M = m.copy()
    for cell in cells:
        i,k = cell
        z = M[i,k]
        newmat = []
        for r in range(m.shape[0]):
            newrow = []
            for c in range(m.shape[1]):
                if i == r:
                    if k == c:  # Pivot cell
                        newrow.append( 1/z )
                    else:       # Pivot row
                        newrow.append( -M[r,c]/z )
                else:
                    if k == c:  # Pivot col
                        newrow.append( M[r,c]/z )
                    else:       # Elsewhere
                        newrow.append( M[r,c] - M[i,c]*M[r,k]/z )
            newmat.append(newrow)
        M =  Matrix(newmat)
    #
    return M

def FullInversion(m):
    "Full matrix inversion is partial inversion at all i==j."
    cells = [(i,i) for i in range(m.shape[0])]
    return sp_partial_inversion(m, *cells)

def inverse_2D(matrix):
    matrix_inv = (1/(matrix[0, 0]*matrix[1, 1] - matrix[0, 1]*matrix[1, 0]))* sym.Matrix([[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]])
    return matrix_inv

Bb, h, s, epsilon = sym.symbols("Bb h s epsilon", real=True)
V, A, B, Cr, D, E, F, G, H, Ii, Cc, Dm = sym.symbols("V A B Cr D E F G H Ii Cc Dm", real=True, constant=True)
a, ad, b, bd, c, cd, e, ed = sym.symbols("a ad b bd c cd e ed", real=True)


#cubic
A  = 1
B  = 1
Cr = 1
D  = 1-h*s
E  = 1-h*s
F  = 1-h*s
G  = 1-s
H  = 1-s
Ii = 1-s

H1 = (ad - ed)*ad*ad*a*a*e
H2 = (1-epsilon)*(ad - ed)*ad*bd*a*b*e/2 + (1+epsilon)*(bd - ed)*ad*bd*a*b*e/2
H3 = (bd - ed)*ad*cd*a*c*e
H4 = (1-epsilon)*(ad - ed)*bd*ad*a*b*e/2 + (1+epsilon)*(bd - ed)*bd*ad*a*b*e/2
H5 = ((1-epsilon)**2)*(ad - ed)*bd*bd*b*b*e/4 + (1-epsilon)*(1+epsilon)*(bd - ed)*bd*bd*b*b*e/2 + ((1+epsilon)**2)*(cd - ed)*bd*bd*b*b*e/4
H6 = (1-epsilon)*(bd - ed)*bd*cd*b*c*e/2 + (1+epsilon)*(cd - ed)*bd*cd*b*c*e/2
H7 = (cd - ed)*cd*cd*c*c*e
H8 = (1-epsilon)*(bd - ed)*bd*cd*c*b*e/2 + (1+epsilon)*(cd - ed)*cd*bd*b*c*e/2
H9 = (bd - ed)*cd*ad*c*a*e

HDmal = ed*a - ad*a + ed*b - bd*b + ed*c - cd*c 

Ht   = -(HDmal*Dm + (Bb/V**2)*(A*H1 + B*H2 + Cr*H3 + D*H4 + E*H5 + F*H6 + G*H7 + H*H8 + Ii*H9)) 



# #seperated birth and drive
# A = 1
# B = 1-h*s
# Cr = 1-s
# D = 1
# E = 1
# F = 1
# G = 1
# H = 1

# H1 = (ad - ed)*ad*a*e
# H2 = (bd - ed)*bd*b*e
# H3 = (cd - ed)*cd*c*e
# H4 = -epsilon*(ad - ed)*ad*bd*a*b*e/2 + epsilon*(bd - ed)*ad*bd*a*b*e/2
# H5 = -epsilon*(ad - ed)*ad*bd*a*b*e/2 + epsilon*(bd - ed)*ad*bd*a*b*e/2
# H6 = -epsilon*(ad - ed)*bd*bd*b*b*e/2 + epsilon*(cd - ed)*bd*bd*b*b*e/2
# H7 = -epsilon*(bd - ed)*cd*bd*c*b*e/2 + epsilon*(cd - ed)*cd*bd*c*b*e/2
# H8 = -epsilon*(bd - ed)*cd*bd*c*b*e/2 + epsilon*(cd - ed)*cd*bd*c*b*e/2

# HDmal = ed*a - ad*a + ed*b - bd*b + ed*c - cd*c 

# Ht   = -(HDmal*Dm + (Bb/V)*((A*H1 + B*H2 + Cr*H3) + (D*H4 + E*H5 + F*H6 + G*H7 + H*H8)/V)) 


aCH, adCH, bCH, bdCH, cCH, cdCH, eCH, edCH = sym.symbols("aCH adCH bCH bdCH cCH cdCH eCH edCH", real=True)
z, Zm, Z, y, Ym, Y, x, Xm, X, t, Tm, T, item, kx, Dx, Tt, r = sym.symbols("z Zm Z y Ym Y x Xm X t Tm T item kx Dx Tt r", real=True)

 
aCH  = z*Zm
adCH = Z
bCH  = y*Ym
bdCH = Y
cCH  = x*Xm
cdCH = X
eCH  = t*Tm
edCH = T

zd, yd, xd, td = sym.symbols("zd yd xd td", real=True)
WW, WD, DD, EE = sym.symbols("WW WD DD EE", real=True)
N1, N2, N3, N4, delta, omega, Dx, kx = sym.symbols("N1 N2 N3 N4 delta omega Dx kx", real=True)

solz = V*WW + (V**0.5)*N1
soly = V*WD + (V**0.5)*N2
solx = V*DD + (V**0.5)*N3
solt = V*EE + (V**0.5)*N4


solZ  = 1 + zd/(V**0.5) + (zd**2)/(2*V)
solZm = 1 - zd/(V**0.5) + (zd**2)/(2*V)
solY  = 1 + yd/(V**0.5) + (yd**2)/(2*V)
solYm = 1 - yd/(V**0.5) + (yd**2)/(2*V)
solX  = 1 + xd/(V**0.5) + (xd**2)/(2*V)
solXm = 1 - xd/(V**0.5) + (xd**2)/(2*V)
solT  = 1 + td/(V**0.5) + (td**2)/(2*V)
solTm = 1 - td/(V**0.5) + (td**2)/(2*V)

Ht = Ht.subs({a: aCH, ad: adCH, b: bCH, bd: bdCH, c:cCH, cd: cdCH, e: eCH, ed: edCH})
Ht = Ht.subs({Z*z*Zm: z-1, Y*y*Ym: y-1, X*x*Xm: x-1, T*t*Tm: t-1})
Ht = Ht.subs({Z: solZ, Zm: solZm, Y: solY, Ym: solYm, X: solX, Xm: solXm, T:solT, Tm:solTm})
Ht = Ht.subs({z: solz, y:soly, x:solx, t:solt})


Ht_MF = Ht/(V**0.5)
Ht_PS = Ht
Ht_ex = Ht*(V**0.5) #**0.5)


L_MF  = sym.collect(sym.expand(Ht_MF), V)
L_PS  = sym.collect(sym.expand(Ht_PS), V)
L_ex  = sym.collect(sym.expand(Ht_ex), V)
Ht_MF = L_MF.coeff(V, 0)
Ht_PS = L_PS.coeff(V, 0)
Ht_ex = L_ex.coeff(V, 0)

source = [WW, WD, DD]
y = [zd, yd, xd]
x = [N1, N2, N3]

Ht_PS = Ht_PS.subs({h: 0.3, Bb: 36, epsilon:0.95, s:0.95, Dm:1, WW: WW/Cc, WD: WD/Cc, DD: DD/Cc, N1:N1/sym.sqrt(Cc), N2:N2/sym.sqrt(Cc), N3:N3/sym.sqrt(Cc)})
Ht_PS = Ht_PS.subs({EE: Cc - WW - WD - DD})


eq1 = -sym.expand(sym.diff(Ht_PS, zd))
eq2 = -sym.expand(sym.diff(Ht_PS, yd))
eq3 = -sym.expand(sym.diff(Ht_PS, xd))
eq4 = -sym.expand(sym.diff(Ht_PS, td))



print(eq1, '\n')
print(eq2, '\n')
print(eq3, '\n')
print(sym.simplify(eq4 - eq1 - eq2 - eq3), '\n')





# Ht_PS = Ht_PS.subs({EE: Cc, h: 0.3, Bb: 36, WD:0, DD:0, WW:0, epsilon:1, s:1, Dm:1, N1:N1/sym.sqrt(Cc), N2:N2/sym.sqrt(Cc), N3:N3/sym.sqrt(Cc)})
# Ht_ex = Ht_ex.subs({h: 0.3, Bb: 36, WD:0, DD:0, WW:0, epsilon:1, s:1, Dm:1, EE:Cc, N1:N1/sym.sqrt(Cc), N2:N2/sym.sqrt(Cc), N3:N3/sym.sqrt(Cc)})

# eq1 = -sym.expand( (sym.diff(Ht_ex, zd)/sym.sqrt(Cc) + sym.diff(Ht_PS, zd) + sym.diff(Ht_ex, zd)/Cc)*sym.sqrt(Cc) )
# eq2 = -sym.expand((sym.diff(Ht_ex, yd)/sym.sqrt(Cc)  + sym.diff(Ht_PS, yd) + sym.diff(Ht_ex, yd)/Cc)*sym.sqrt(Cc))
# eq3 = -sym.expand((sym.diff(Ht_ex, xd)/sym.sqrt(Cc)  + sym.diff(Ht_PS, xd) + sym.diff(Ht_ex, xd)/Cc)*sym.sqrt(Cc))
# eq4 = -sym.expand((sym.diff(Ht_ex, td)/sym.sqrt(Cc)  + sym.diff(Ht_PS, td) + sym.diff(Ht_ex, td)/Cc)*sym.sqrt(Cc))


# print(eq1, '\n')
# print(eq2, '\n')
# print(eq3, '\n')
# print(sym.simplify(eq4 + eq1 + eq2 + eq3), '\n')


# eq1 = sym.simplify((sym.diff(Ht_ex, N1)/sym.sqrt(Cc)  + sym.diff(Ht_PS, N1))*sym.sqrt(Cc))
# eq2 = sym.simplify((sym.diff(Ht_ex, N2)/sym.sqrt(Cc)  + sym.diff(Ht_PS, N2))*sym.sqrt(Cc))
# eq3 = sym.simplify((sym.diff(Ht_ex, N3)/sym.sqrt(Cc)  + sym.diff(Ht_PS, N3))*sym.sqrt(Cc))
# eq4 = sym.simplify((sym.diff(Ht_ex, N4)/sym.sqrt(Cc)  + sym.diff(Ht_PS, N4))*sym.sqrt(Cc))


# print(eq1, '\n')
# print(eq2, '\n')
# print(eq3, '\n')
# print(sym.simplify(eq4), '\n')

# #print(sym.solve([eq1, eq2, eq3], [N1, N2, N3]))





#TO CALCULATE HAMILTONIAN USING FUNCTIONS:
    
# def calc_fit(I1):
    
#     if I1[0] == bd or I1[1] == bd:
#         if (I1[0] == bd and I1[1] == bd): #or (I1[0] == bd and I1[1] == bd):
#             fit = 1-s
#         else:
#             fit = 1-h*s
#     else:
#         fit = 1
        
#     return fit


# def calc_rate(I1, I2, Out):
    
#     R = []

#     av_fit = calc_fit(I1)
    
#     for O1 in Out:
        
        
#         if I1[0] == bd or I1[1] == bd:

#             if I1[0] == bd and I1[1] == bd:
#                 multi1 = 1
                  
#             else:
#                 if O1[0] == bd:
#                     multi1 = (1+epsilon)
#                 else: 
#                     multi1 = (1-epsilon)
#         else:
#             multi1 = 1
            
            
            
#         if I2[0] == bd or I2[1] == bd:

#             if I2[0] == bd and I2[1] == bd:
#                 multi2 = 1
                  
#             else:
#                 if O1[1] == bd:
#                     multi2 = (1+epsilon)
#                 else:
#                     multi2 = (1-epsilon)
#         else:
#             multi2 = 1
                    
#         multi = multi1*multi2
#         rate  = av_fit*multi/4
#         R.append(rate)
        
#     return R


# def H_part(I1, I2, Ibar1, Ibar2, Out):
#     H = 0
#     Rate = calc_rate(I1, I2, Out)
    
#     for i in range(0, len(Out)):
#         O1 = Out[i]
#         R  = Rate[i]
#         H += R*(O1[0]*O1[1] - ed)*(I1[0]*I1[1])*(I2[0]*I2[1])*(Ibar1[0]*Ibar1[1])*(Ibar2[0]*Ibar2[1])*e
#     return Bb*H/(V**4)


# def H_void(geno_crea, geno_anih):
#     H_death = 0
    
#     for i in range(0, len(geno_crea)):
#         X    = geno_crea[i]
#         Xbar = geno_anih[i]
#         H_death += ed*(Xbar[0]*Xbar[1]) - X[0]*X[1]*Xbar[0]*Xbar[1]
#     return H_death/V


# Bb, h, s, epsilon, V, Cc, Dm = sym.symbols("Bb h s epsilon V Cc Dm", real=True)
# a, ad, b, bd, c, cd, e, ed = sym.symbols("a ad b bd c cd e ed", real=True)

# crea = [ad, bd]
# anih = [a, b]

# geno_crea = list(itertools.combinations(crea,2)) + list(zip(crea, crea))
# geno_anih = list(itertools.combinations(anih,2)) + list(zip(anih, anih))


# H_death = H_void(geno_crea, geno_anih)

# #sym.pprint(H_death)

# H_full  = H_death*Dm

# for i in range(0, len(geno_crea)):
#     for j in range(0, len(geno_crea)):
#         I1     = geno_crea[i]
#         #print(I1)
#         I2     = geno_crea[j]
#         #print(I2)
#         Ibar1  = geno_anih[i]
#         Ibar2  = geno_anih[j]
#         Out    = [(x, y) for x, y in itertools.product(I1, I2)]
#         Outbar = [(x, y) for x, y in itertools.product(Ibar1, Ibar2)]
        
#         H_full += H_part(I1, I2, Ibar1, Ibar2, Out)




                       

                       
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:40:48 2024

@author: corneliasheeran
"""

import sympy as sym
import numpy as np
from sympy import *
import itertools

Bb, h, s, epsilon, V, Cc, Dm, nu, hN = sym.symbols("Bb h s epsilon V Cc Dm nu hN", real=True)
a, ad, b, bd, c, cd, e, ed = sym.symbols("a ad b bd c cd e ed", real=True)


def calc_fit(I1):
    
    if (I1[0] == ad or I1[1] == ad): #one W allele
        if (I1[0] == bd or I1[1] == bd): #WD
            fit = 1-h*s
        elif (I1[0] == cd or I1[1] == cd): #WN
            fit = 1-hN*s
        else:
            fit = 1 #WW
    else:
        fit = 1-s #NN, DN, DD
        
    return fit


def calc_rate(I1, I2, Out):
    
    R = []

    av_fit = calc_fit(I1)
    
    for O1 in Out:
        
        if O1[0] == cd or O1[1] == cd:
            rate = sym.simplify(av_fit/4)
            R.append(rate)
        else:
            
            if I1[0] == cd or I1[1] == cd: #1 N allele
                multi1 = 1
            elif I1[0] == bd or I1[1] == bd: #if one input allele is drive
    
                if I1[0] == bd and I1[1] == bd: #if both input alleles are drive
                    multi1 = 1
                      
                else:
                    if O1[0] == bd:
                        multi1 = (1+epsilon)
                    else: 
                        multi1 = (1-epsilon)
            else:
                multi1 = 1
                
                
            if I2[0] == cd or I2[1] == cd: #1 N allele
                multi2 = 1
                
            elif I2[0] == bd or I2[1] == bd:
    
                if I2[0] == bd and I2[1] == bd:
                    multi2 = 1
                      
                else:
                    if O1[1] == bd:
                        multi2 = (1+epsilon)
                    else:
                        multi2 = (1-epsilon)
            else:
                multi2 = 1
                        
            multi = multi1*multi2
            rate  = sym.simplify(av_fit*multi/4)
            R.append(rate)
        
    return R, av_fit


def H_part(I1, I2, Ibar1, Ibar2, Out):
    H = 0
    Rate, av_fit = calc_rate(I1, I2, Out)
    
    for i in range(0, len(Out)):
        O1 = Out[i]
        R  = Rate[i]
        R4 = sym.simplify(R*4)
        
        if '(1 + epsilon)' in str(R4) or '(epsilon + 1)' in str(R4) or 'epsilon + 1' in str(R4) or '1 + epsilon' in str(R4):
            R  = (R/(1+epsilon))*(1+epsilon*(1-nu))
            H += R*(O1[0]*O1[1] - ed)*(I1[0]*I1[1])*(I2[0]*I2[1])*(Ibar1[0]*Ibar1[1])*(Ibar2[0]*Ibar2[1])*e
            
            R1 = av_fit*epsilon*nu/4
            
            if (O1[0] == bd and O1[1] == bd):
                O2 = (bd, cd)
            else:
                O2 = (ad, cd)
                
            H += R1*(O2[0]*O2[1] - ed)*(I1[0]*I1[1])*(I2[0]*I2[1])*(Ibar1[0]*Ibar1[1])*(Ibar2[0]*Ibar2[1])*e
        
        else:
            H += R*(O1[0]*O1[1] - ed)*(I1[0]*I1[1])*(I2[0]*I2[1])*(Ibar1[0]*Ibar1[1])*(Ibar2[0]*Ibar2[1])*e

    return Bb*H/(V**4)


def H_void(geno_crea, geno_anih):
    H_death = 0
    
    for i in range(0, len(geno_crea)):
        X    = geno_crea[i]
        Xbar = geno_anih[i]
        H_death += ed*(Xbar[0]*Xbar[1]) - X[0]*X[1]*Xbar[0]*Xbar[1]
    return H_death/V


crea = [ad, bd, cd]
anih = [a, b, c]

geno_crea = list(itertools.combinations(crea,2)) + list(zip(crea, crea))
geno_anih = list(itertools.combinations(anih,2)) + list(zip(anih, anih))


H_death = H_void(geno_crea, geno_anih)

#sym.pprint(H_death)
Ht  = -H_death*Dm

for i in range(0, len(geno_crea)):
    for j in range(0, len(geno_crea)):
        
        I1      = geno_crea[i]
        #print(I1)
        I2      = geno_crea[j]
        #print(I2)
        Ibar1   = geno_anih[i]
        Ibar2   = geno_anih[j]
        Out     = [(x, y) for x, y in itertools.product(I1, I2)]
        #print(Out)
        Outbar  = [(x, y) for x, y in itertools.product(Ibar1, Ibar2)]
        
        #print(H_part(I1, I2, Ibar1, Ibar2, Out))
        Ht += -H_part(I1, I2, Ibar1, Ibar2, Out)
        
        #print('\n')

#sym.pprint(sym.diff(H_full, bd))

aCH, adCH, bCH, bdCH, cCH, cdCH, eCH, edCH = sym.symbols("aCH adCH bCH bdCH cCH cdCH eCH edCH", real=True)
z, Zm, Z, y, Ym, Y, x, Xm, X, t, Tm, T, Tt = sym.symbols("z Zm Z y Ym Y x Xm X t Tm T Tt", real=True)

aCH  = z*Zm
adCH = Z
bCH  = y*Ym
bdCH = Y
cCH  = x*Xm
cdCH = X
eCH  = t*Tm
edCH = T

zd, yd, xd, td = sym.symbols("zd yd xd td", real=True)
W, D, N, Es    = sym.symbols("W D N Es", real=True)
n1, n2, n3, n4 = sym.symbols("n1 n2 n3 n4", real=True)

solz = V*W  + (V**0.5)*n1
soly = V*D  + (V**0.5)*n2
solx = V*N  + (V**0.5)*n3
solt = V*Es + (V**0.5)*n4


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
Ht_ex = Ht*(V**0.5)


L_MF  = sym.collect(sym.expand(Ht_MF), V)
L_PS  = sym.collect(sym.expand(Ht_PS), V)
# L_ex  = sym.collect(sym.expand(Ht_ex), V)
Ht_MF = L_MF.coeff(V, 0)
Ht_PS = L_PS.coeff(V, 0)
# Ht_ex = L_ex.coeff(V, 0)


#print(sym.simplify(sym.expand(sym.diff(Ht_MF, zd) + sym.diff(Ht_MF, yd) + sym.diff(Ht_MF, xd) + 2*sym.diff(Ht_MF, td))))

Wsol = sym.diff(Ht, zd)
Dsol = sym.diff(Ht, yd)

Wsol = sym.simplify(Wsol.subs({N:0, epsilon:1, nu:0}))
Dsol = sym.simplify(Dsol.subs({N:0, epsilon:1, nu:0}))

Ap1 = sym.diff(Ht_PS, zd).subs({zd:Tt, yd:Tt, N:0, epsilon:1, nu:0})
Ap2 = sym.diff(Ht_PS, yd).subs({zd:Tt, yd:Tt, N:0, epsilon:1, nu:0})
Ap3 = sym.diff(Ht_PS, n1).subs({zd:Tt, yd:Tt, N:0, epsilon:1, nu:0})
Ap4 = sym.diff(Ht_PS, n2).subs({zd:Tt, yd:Tt, N:0, epsilon:1, nu:0})

sym.pprint(Ap1)
print('\n')
sym.pprint(Ap2)
print('\n')
sym.pprint(Ap3)
print('\n')
sym.pprint(Ap4)



                       
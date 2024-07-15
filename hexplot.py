#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:32:14 2024

@author: corneliasheeran
"""

import numpy as np
import seaborn as sns
#sns.set_theme(style="ticks")
import matplotlib.pyplot as plt

import h5py

#Load the array from the HDF5 file
with h5py.File("/Users/corneliasheeran/Documents/TM_Connie/drivein.h5", "r") as f:
    drivein = f["dataset"][:]

with h5py.File("/Users/corneliasheeran/Documents/TM_Connie/wildin.h5", "r") as f:
    wildin = f["dataset"][:]

# with h5py.File("Nalldat.h5", "r") as f:
#     wildin = f["dats"][:]
    
def createplots(Tbegin, Tend, step, drivein, title):
    for T in np.arange(Tbegin, Tend, step):
    
        Tdrive = drivein[T, :, :]
        
        X, Y = np.meshgrid(range(Tdrive.shape[0]), range(Tdrive.shape[-1]))
        X, Y = X*2, Y*2
        
        # Turn this into a hexagonal grid
        for i, k in enumerate(X):
            if i % 2 == 1:
                X[i] += 1
                Y[:,i] += 1
        
        figin, ax = plt.subplots()
        im = ax.hexbin(
            X.reshape(-1), 
            Y.reshape(-1), 
            C=Tdrive.reshape(-1), 
            gridsize=int(Tdrive.shape[0]/2)) #, vmin=2, vmax=200)
        
        # the rest of the code is adjustable for best output
        ax.set_aspect(0.8)
        ax.set(xlim=(-4, X.max()+4,), ylim=(-4, Y.max()+4))
        ax.axis(False)
        plt.colorbar(im)
        plt.title(f'{title} at T={T}')
        plt.show()
        plt.clf
        
createplots(50, 100, 5, wildin+drivein, "Total Drive Population")
#createplots(990, 999, 2, wildin, "Total Wild-Type Population")



 
# from matplotlib.animation import FuncAnimation  
   
# figgif = plt.figure()
# # data = np.random.rand(10, 10)
# # sns.heatmap(data, vmax=.8, square=True)
# shape = int(drivein.shape[1]/2)
# square = drivein.shape[1]

# def init():
    
#     X, Y = np.meshgrid(range(square), range(square))
#     X, Y = X*2, Y*2
        
#     # Turn this into a hexagonal grid
#     for i, k in enumerate(X):
#         if i % 2 == 1:
#             X[i] += 1
#             Y[:,i] += 1
            
#     figin, ax = plt.subplots()
#     im = ax.hexbin(
#         X.reshape(-1), 
#         Y.reshape(-1), 
#         C=np.zeros((square, square)).reshape(-1), 
#         gridsize=shape)
#       #sns.heatmap(np.zeros((10, 10)), vmax=.8, square=True, cbar=False)

# def animate(i):
#     data = drivein[i, :, :]
#     X, Y = np.meshgrid(range(data.shape[0]), range(data.shape[-1]))
#     X, Y = X*2, Y*2
        
#     # Turn this into a hexagonal grid
#     for j, k in enumerate(X):
#         if j % 2 == 1:
#             X[j] += 1
#             Y[:,j] += 1
        
#     figin, ax = plt.subplots()
#     im = ax.hexbin(
#         X.reshape(-1), 
#         Y.reshape(-1), 
#         C=data.reshape(-1), 
#         gridsize=int(data.shape[0]/2))
        
#         # the rest of the code is adjustable for best output
#     #sns.heatmap(data, vmax=.8, square=True, cbar=False)

# anim = FuncAnimation(figgif, animate, init_func=init, frames=100, repeat=False)
# plt.show()
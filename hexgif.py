#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:57:02 2024

@author: corneliasheeran
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py

# with h5py.File("drivein.h5", "r") as f:
#     drivein = f["dataset"][:]

# with h5py.File("wildin.h5", "r") as f:
#     wildin = f["dataset"][:]
    
with h5py.File("Nalldat3.h5", "r") as f:
    wildin = f["dats"][:]


def create_hexbin_plot(T, drivein, ax):
    Tdrive = drivein[:, :, T]

    X, Y = np.meshgrid(range(Tdrive.shape[0]), range(Tdrive.shape[1]))
    X, Y = X * 2, Y * 2

    # Turn this into a hexagonal grid
    for i, k in enumerate(X):
        if i % 2 == 1:
            X[i] += 1
            Y[:, i] += 1

    im = ax.hexbin(
        X.reshape(-1), 
        Y.reshape(-1), 
        C=Tdrive.reshape(-1), 
        gridsize=int(Tdrive.shape[0] / 2), vmin=0, vmax=200)
    

    # the rest of the code is adjustable for best output
    ax.set_aspect(0.8)
    #ax.set(xlim=(2, 100), ylim=(2, 100))
    ax.set(xlim=(-4, X.max() + 4), ylim=(-4, Y.max() + 4))
    ax.axis(False)
    #cb = plt.colorbar(im, ax=ax)

    return im


def init():
    return []

def update(T, drivein, ax):
    ax.clear()
    #cb.ax.clear()
    create_hexbin_plot(T, drivein, ax)
    return []

def create_animation(Tbegin, Tend, step, drivein, interval=100, save_path=None):
    fig, ax = plt.subplots()
    # data = np.random.rand(10, 10)
    # cax = ax.imshow(data, cmap='viridis')
    # cb = fig.colorbar(cax)

    anim = FuncAnimation(
        fig, update, fargs=(drivein, ax), frames=np.arange(Tbegin, Tend, step), 
        init_func=init, interval=interval, blit=False
    )

    if save_path:
        anim.save(save_path, writer='ffmpeg')
    
    plt.show()


Tbegin = 0
Tend = 30000
step = 100

create_animation(Tbegin, Tend, step, wildin[2, :, :, :], interval=50, save_path='2sig=3.mp4')

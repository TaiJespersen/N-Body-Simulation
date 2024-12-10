#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:19:13 2024

@author: tai
"""

import random
import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
plt.gca().set_aspect('equal')
plt.rcParams['figure.dpi'] = 300
plt.figure(figsize=(8, 8))

Size = 1000

Xs = []
Ys = []
for i in range(1000):
    theta= random.random() * 2 * np.pi
    random_value = random.random()
    
    x,y = ( ( Size * sqrt(random_value) ) *np.cos(theta) ), ( ( Size * sqrt(random_value) ) *np.sin(theta) )
    Xs.append(x)
    Ys.append(y)

plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)

circle = plt.Circle((0, 0), Size, color='blue', fill=False, linestyle='-', linewidth=2, label='Circle with Radius S')
plt.gca().add_artist(circle)




plt.xlim(-1500,1500)
plt.ylim(-1500,1500)
plt.title("Particle Generation")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.scatter(Xs,Ys, alpha=1, color='purple', label='Particles', s=30)

plt.legend(loc='upper right', fontsize=12)
plt.show()

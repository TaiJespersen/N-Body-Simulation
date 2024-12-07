#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:06:22 2024

@author: tai
"""

import numpy as np
import matplotlib.pyplot as plt

data = [
    (5.1, 0.0), (4.9, 0.1), (4.8, 0.7), (5.0, 1.3), (4.5, 1.8),
    (3.5, 2.0), (3.3, 2.0), (3.8, 2.6), (2.8, 2.2), (2.0, 2.8),
    (1.4, 3.0), (0.8, 3.1), (0.4, 2.7), (-0.7, 3.1), (-0.9, 2.8),
    (-1.6, 2.8), (-2.6, 2.4), (-3.0, 2.4), (-3.4, 2.0), (-3.8, 2.2),
    (-4.2, 2.1), (-4.3, 1.1), (-5.0, 1.2), (-5.0, 0.4), (-4.6, 0.2),
    (-5.2, -0.1), (-4.5, -0.8), (-5.0, -0.9), (-4.1, -1.3), (-4.1, -1.7),
    (-3.7, -1.9), (-3.7, -2.3), (-2.9, -2.6), (-2.2, -2.5), (-1.6, -2.8),
    (-0.9, -3.0), (-0.4, -3.0), (-0.1, -2.8), (0.7, -3.2), (1.2, -2.6),
    (1.9, -2.6), (2.6, -2.5), (3.1, -2.3), (3.7, -2.2), (3.7, -1.4),
    (4.1, -1.6), (4.4, -1.2), (4.9, -0.9), (4.9, -0.7), (5.1, 0.0)
]

x = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])


A = np.stack([x**2, x * y, y**2, x, y]).T
b = np.ones_like(x)
w = np.linalg.lstsq(A, b)[0].squeeze()

xlin = np.linspace(-10, 10, 300)
ylin = np.linspace(-10, 10, 300)
X, Y = np.meshgrid(xlin, ylin)

Z = w[0]*X**2 + w[1]*X*Y + w[2]*Y**2 + w[3]*X + w[4]*Y

fig, axe = plt.subplots()
axe.scatter(x, y)
axe.contour(X, Y, Z, [1])
axe.show()
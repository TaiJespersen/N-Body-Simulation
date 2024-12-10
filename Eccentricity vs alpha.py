#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:14:09 2024

@author: tai
"""
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


plotX = [0,	0.00000025,	0.0000005,	0.00000075,	0.000000875,	0.0000009375,	0.000001,	0.0000015]

plotY = [0.8396842878,	0.8368947249,	0.8035676713,	0.7725503447,	0.7460361803,	0.735649507,	0.7292035218,	0.6793606297]

#ECCENTRICITY

plt.figure(figsize=(12, 6))
plt.grid(False, which='both', linestyle='--', color='gray', alpha=0.5)
plt.title('Eccentricity vs α', fontsize=18, fontweight='bold')
plt.xlabel('α (Initial Angular Velocity)', fontsize=14)
plt.ylabel('Average Eccentricity', fontsize=14)

plt.scatter(plotX, plotY, color="darkblue", s=100, marker='s')
plt.show()


plotX2 = [0,	0.00000025,	0.0000005,	0.00000075,	0.000000875,	0.0000009375,	0.000001,	0.0000015]

plotY2 = [3674,	3395,	3748,	3928,	4056,	3770,	2661,	1167]

#ECCENTRICITY

plt.figure(figsize=(12, 6))
plt.grid(False, which='both', linestyle='--', color='gray', alpha=0.5)
plt.title('Orbit Number vs α', fontsize=18, fontweight='bold')
plt.xlabel('α (Initial Angular Velocity)', fontsize=14)
plt.ylabel('Orbit Number', fontsize=14)

plt.scatter(plotX2, plotY2, color="darkblue", s=100, marker='s')
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:10:54 2024

@author: tai
"""


import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

from scipy import constants
from numpy import sqrt, linspace, histogram2d, zeros, digitize, arctan, pi
import numpy as np
from numpy.linalg import eig

plt.rcParams['figure.dpi'] = 300

TIME = 0

# Constants for quadtree configuration
MAX_POINTS = 2  # Maximum points per node before splitting
MIN_NODE_SIZE = 1  # Minimum size of a node before it stops subdividing

NumPoints = 1000 # Number of generated particles
MaxMass = 50000  # Maximum particle mass
MaxVelocity = 0.0    # Maximum particle velocity
AngularVelocity = 0.0000009375
Size = 1000     # size of window
#h = 10  # smoothing distance

totTime = 33 * (20000000/33)
resolution = 1 * (10 ** 4) # lower time resolution = better
#Fres = 10 # higher force resolution = better
Inter = 1 #Animation Inteval
θ = 0.5  #Barnes Hut number - lower = better resolution
overlap_dist = Size/5    #Max Distance to count particles as overlapping
ε=100


G = constants.G

t = resolution

class Point:
    def __init__(self, position, mass, velocityI, velocityJ):
        
        self.position = (position)  # (x, y) position of the point
        self.mass = mass          # Mass of the point
        self.force = (0.0, 0.0)   # Initialize force as a tuple (Fx, Fy)
        
        self.angularVelocity = AngularVelocity
        
        #angular velocity
        self.velocity = (-self.angularVelocity * self.position[1], self.angularVelocity * self.position[0])
        
        #regular velocity
        #self.velocity = (velocityI, velocityJ)
        
        self.θ = (np.arctan2(position[1], position[0]))

        self.startingTheta = np.arctan2(position[1],position[0])
        self.startingTheta += (np.pi)
        
        self.orbitting = 0
        self.eccentricity = []
        
        self.overlaps = (0.0)        
        self.orbit = []
        


class QuadTreeNode:
    def __init__(self, boundary):
        """Boundary is a tuple: (x_center, y_center, half_size)"""
        self.boundary = boundary  # (x_center, y_center, half_size)
        self.points = []          # List of points in this node
        self.children = []        # Child nodes (if any)
        self.center_of_mass = (0, 0)  # Center of mass of the points in this node
        self.mass = 0             # Total mass of points in this node

    def is_leaf(self):
        """Check if the node is a leaf (i.e., has no children)."""
        return len(self.children) == 0

    def subdivide(self):
        """Subdivide the node into 4 children."""
        x_center, y_center, half_size = self.boundary
        
        # Create boundaries for the four quadrants
        self.children = [
            QuadTreeNode((x_center - half_size / 2, y_center - half_size / 2, half_size / 2)),  # Bottom left
            QuadTreeNode((x_center + half_size / 2, y_center - half_size / 2, half_size / 2)),  # Bottom right
            QuadTreeNode((x_center - half_size / 2, y_center + half_size / 2, half_size / 2)),  # Top left
            QuadTreeNode((x_center + half_size / 2, y_center + half_size / 2, half_size / 2)),  # Top right
        ]
    def insert(self, point):
        """Insert a point into the quadtree."""

        if not self.contains(point):
            return  # If point is out of bounds, do nothing
        
        # Handle case when leaf and within MAX_POINTS or MIN_NODE_SIZE
        if len(self.points) >= MAX_POINTS and self.boundary[2] < MIN_NODE_SIZE:
            if self.is_leaf():
                self.subdivide()
                for p in self.points:
                    for child in self.children:
                        if child.contains(p):
                            child.insert(p)
                self.points = []
        else:
            self.points.append(point)
            self.mass += point.mass
            self.update_center_of_mass(point)

    
    def contains(self, point):
        """Check if the point is within the node's boundary."""
        x_center, y_center, half_size = self.boundary
        x, y = point.position
        return (x_center - half_size <= x <= x_center + half_size and
                y_center - half_size <= y <= y_center + half_size)

    def update_center_of_mass(self, point):
        """Update the center of mass for this node."""
        total_mass = self.mass + point.mass  # Update total mass by adding the new point's mass
        if total_mass > 0:  # Avoid division by zero
            # Update the center of mass based on the new point
            self.center_of_mass = (
                (self.center_of_mass[0] * self.mass + point.position[0] * point.mass) / total_mass,
                (self.center_of_mass[1] * self.mass + point.position[1] * point.mass) / total_mass
            )
            self.mass = total_mass  # Now set the new total mass for the node


    def calculate_forces(self, point):           
        #print("CALC FORCES")
        # If the node is a leaf and contains points, calculate forces directly
        if self.is_leaf() and len(self.points) > 0:
            #print(f'LEAF SIZE = {len(self.points)}')
            for other_point in self.points:
                if other_point != point:  # Avoid self-interaction
                    self.apply_gravity(point, other_point)
                    #print("APPLYING GRAV1")
        else:
            #print("IS NOT LEAF")
            # Calculate the distance between the point and the node's center of mass
            x, y = point.position
            x_cm, y_cm = self.center_of_mass
            distance_to_cm = sqrt((x - x_cm)**2 + (y - y_cm)**2)
            #print(f'Distance to center of mass: {distance_to_cm}, Boundary size: {self.boundary[2]}')
            # Barnes-Hut criterion: if the node is far enough, approximate using the center of mass
            if self.boundary[2] / distance_to_cm < θ:  # θ is the Barnes-Hut threshold
                #print(f'Distance to center of mass: {distance_to_cm}, Boundary size: {self.boundary[2]}')
                if self.mass > 0:  # Avoid division by zero if there's no mass in the node
                    #print("APPLYING GRAV2")
                    self.apply_gravity(point, Point(self.center_of_mass, self.mass))
            else:
                # Otherwise, recurse into the children nodes for more detailed calculations
                for child in self.children:
                    if child.points or not child.is_leaf():  # Recurse only into non-empty children
                        child.calculate_forces(point)
                        #print("RECURSE")
                

    def should_check(self, node, point):
        """Check if the node should be checked based on the distance."""
        x, y = point.position
        x_center, y_center, half_size = node.boundary
        distance_to_center = sqrt((x - x_center)**2 + (y - y_center)**2)
        
        
        #distance_to_center < half_size * 2  # Example threshold
        # Adjust the condition to include other factors, such as half_size or mass
        return half_size / distance_to_center < θ

    

    def apply_gravity(self, point, other_point):
        di = other_point.position[0] - point.position[0]
        dj = other_point.position[1] - point.position[1]
        r = sqrt(di**2 + dj**2)
        
        if r <= overlap_dist:
            point.overlaps += 1
            '''
        def g(r):
            def W2(u):
                if 0 <= u < 1/2:
                    r1 = (16/3) * (u**2)
                    r2 = (-48/5) * (u**4)
                    r3 = (32/5) * (u**5)
                    r4 = (-14/5)
                    return r1 + r2 + r3 + r4
                if 1/2 <= u < 1:
                    r1 = (1/(15 * u))
                    r2 = (32/3) * (u**2)
                    r3 = -16 * (u**3)
                    r4 = (48/5) * (u**4)
                    return r1 + r2 + r3 + r4
                if u >= 1:
                    return -1/u
            return (-1/h) * W2(r/h) * (r/h)
        '''

        #F = (G * point.mass * other_point.mass) * g(r)
        F = (G * point.mass * other_point.mass) / (r**2 + ε**2)
        Fi = F * di / r  # Force in the x-direction
        Fj = F * dj / r  # Force in the y-direction

        # Update the force for the point
        point.force = (point.force[0] + Fi, point.force[1] + Fj)
        #other_point.force = (other_point.force[0] - Fi, other_point.force[1] - Fj)

        

class QuadTree:
    def __init__(self, boundary):
        self.root = QuadTreeNode(boundary)

    def insert(self, point):
        """Insert a point into the quadtree starting from the root."""
        self.root.insert(point)

    def calculate_forces(self):
        """Calculate the gravitational forces on each point."""
        for point in self.root.points:
            self.root.calculate_forces(point)  # Calculate forces for each point
    def contains_point(self, point):
        """Check if the point is within the quadtree's root boundary."""
        return self.root.contains(point)
            
    
            
            
            
            
            
            
            
            
            
            



# Initialize the quadtree
quadtree_size = Size
quadtree_center = (0, 0)  # Center of the quadtree
boundary = (quadtree_center[0], quadtree_center[1], quadtree_size / 2)  # (x_center, y_center, half_size)
quadtree = QuadTree(boundary)

points = [
    Point(
        position=( 
            ( ( Size * sqrt(random_value) ) *np.cos(theta) ), 
            ( ( Size * sqrt(random_value) ) *np.sin(theta) ) 
            ),  # Random position within the range
        mass = random.random() * MaxMass,
        velocityI = 0,#(random.random() * MaxVelocity) - (0.5 * MaxVelocity),
        velocityJ = 0#(random.random() * MaxVelocity) - (0.5 * MaxVelocity)
    ) for _ in range(NumPoints)
    for random_value, theta in [(random.random(), random.random() * 2 * np.pi)]
]

for point in points:
    quadtree.insert(point)

def ellipsefit(orbit):
    '''
    xs = np.array([i[0] for i in orbit])
    ys = np.array([i[1] for i in orbit])
    
    fig, axe = plt.subplots()
    axe.scatter(xs, ys)
    plt.show()
    '''
    '''
    x = np.array([i[0] for i in orbit])
    y = np.array([i[1] for i in orbit])
    
    A = np.stack([x**2, x * y, y**2,x , y]).T
    b = np.ones_like(x)
    w = np.linalg.lstsq(A, b)[0].squeeze()
    
    A, B, C, D, E = w
    
    # Form the matrix M
    M = np.array([[A, B / 2], [B / 2, C]])
    
    # Find the eigenvalues and eigenvectors of M to determine the ellipse axes and orientation
    eigvals, eigvecs = eig(M)  # Only the upper-left 2x2 part matters for the ellipse
    print(f"EIGVALS {eigvals}")
    # Sort eigenvalues (larger eigenvalue is the semi-major axis)
    eigvals = np.abs(eigvals)  # Eigenvalues should be positive, so take absolute values
    eigvals = np.sort(eigvals)
    
    print(f"EIGVALS2 {eigvals}")
    a = np.sqrt(1 / eigvals[0])  # Semi-major axis (larger eigenvalue)
    b = np.sqrt(1 / eigvals[1])  # Semi-minor axis (smaller eigenvalue)
    print(f"A {A} B {B}")
    # Compute the eccentricity of the ellipse
    eccentricity = np.sqrt(1 - (b**2 / a**2))# Example data points (x, y)
    print(f"ECCENTRICITY {eccentricity}")
    
    
    xlin = np.linspace(-1000, 1000, 300)
    ylin = np.linspace(-1000, 1000, 300)
    X, Y = np.meshgrid(xlin, ylin)
    
    Z = w[0]*X**2 + w[1]*X*Y + w[2]*Y**2 #+ w[3]*X + w[4]*Y
    
    
    fig, axe = plt.subplots(figsize=(8, 6))
    plt.grid(False, which='both', linestyle='--', color='gray', alpha=0.5)
    axe.scatter(x, y, label='Particle Orbit', color='firebrick')
    contour = axe.contour(X, Y, Z, [1])
    
    from matplotlib.lines import Line2D
    
    # Create custom legend entry for the contour line
    ellipse_line = Line2D([0], [0], color='purple', lw=2)  # Adjust color/lw as needed
    axe.legend([ellipse_line, plt.Line2D([], [], marker='o', color='firebrick', label='Particle Orbit')],
               ['Fitted Ellipse', 'Particle Orbit'], loc='upper right')
    plt.show()
    
    '''    
    
    
    
    
    
    
    
    if abs( max(orbit[0]) ) < 100 and abs( max(orbit[1]) ) < 100:
        return np.nan
    
    x = np.array([i[0] for i in orbit])
    y = np.array([i[1] for i in orbit])
    
    A = np.stack([x**2, x * y, y**2,x , y]).T
    b = np.ones_like(x)
    w = np.linalg.lstsq(A, b)[0].squeeze()
    
    A, B, C, D, E = w

    # Form the matrix M
    M = np.array([[A, B / 2], [B / 2, C]])
    
    # Find the eigenvalues and eigenvectors of M to determine the ellipse axes and orientation
    eigvals, eigvecs = eig(M)  # Only the upper-left 2x2 part matters for the ellipse
    # Sort eigenvalues (larger eigenvalue is the semi-major axis)
    eigvals = np.abs(eigvals)  # Eigenvalues should be positive, so take absolute values
    eigvals = np.sort(eigvals)
    
    a = np.sqrt(1 / eigvals[0])  # Semi-major axis (larger eigenvalue)
    b = np.sqrt(1 / eigvals[1])  # Semi-minor axis (smaller eigenvalue)
    # Compute the eccentricity of the ellipse
    eccentricity = np.sqrt(1 - (b**2 / a**2))# Example data points (x, y)
    return eccentricity     

    

def timestep():
    
    global quadtree
    global TIME
    
    quadtree = QuadTree(boundary)
    for point in points:
        quadtree.insert(point)

    # Calculate forces for each point
    for point in points:
        quadtree.root.calculate_forces(point)
    
    for point in points:
        # Calculate acceleration from forces
        ax = point.force[0] / point.mass
        ay = point.force[1] / point.mass
        
        #print(f'point: {id(point)} X: {point.force[1]} Y: {point.force[1]}')
        
        # Update velocity
        point.velocity = (point.velocity[0] + ax * t, point.velocity[1] + ay * t)
        
        # Update position
        point.position = (
            point.position[0] + point.velocity[0] * t,
            point.position[1] + point.velocity[1] * t
        )
        point.θ = np.arctan2(point.position[1], point.position[0])
        point.θ += (np.pi)
        
        point.orbit.append(point.position)
        
        if point.orbitting == 0:
            if abs(point.θ - point.startingTheta) >= pi:
                point.orbitting = 1
            else:
                pass
        elif point.orbitting == 1:
            if abs(point.θ - point.startingTheta) < 0.1:
                point.eccentricity.append([ellipsefit(point.orbit), TIME])
                point.orbit = []
                point.orbitting = 0
            else:
                pass
    # Reset forces for the next timestep
    for point in points:
        point.force = (0, 0)


fig, ax = plt.subplots()
fig.patch.set_facecolor('silver')
ax.set_facecolor('silver')
ax.set_aspect('equal')
scatter = ax.scatter([p.position[0] for p in points], [p.position[1] for p in points], alpha = 1, s = [p.mass / 1000 for p in points])
ax.set_xlim(-2 * Size, Size * 2)
ax.set_ylim(-2 * Size, Size * 2)


def update_colors():
    overlaps = [p.overlaps for p in points]
    max_overlap = NumPoints

    normalized_overlaps = [overlap / max_overlap for overlap in overlaps]  # Normalize overlaps
    
    colors = plt.cm.viridis(normalized_overlaps)
    return colors


def update(frame):
    global TIME
    TIME += 1
    
    if ( (TIME * resolution) // totTime) % .01 == 0:
        print(f'{(100* TIME * resolution)/totTime}% complete')
    timestep()
    
    
    x_positions = [p.position[0] for p in points]
    y_positions = [p.position[1] for p in points]
    
    # Update scatter plot
    ax.set_title(f'Time: {TIME}')
    scatter.set_offsets(list(zip(x_positions, y_positions)))  
    scatter.set_facecolor(update_colors())
    for point in points:
        point.overlaps = 0
    
    return (scatter, )


frames=int(totTime / resolution)
    
ani = FuncAnimation(fig, update, frames, blit = True, interval = Inter)

Writer = writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=4000)

ani.save("filename.mp4", writer=writer)

plt.show()

def Eccentricities():
    Eccentricities = []
    Times = []
    for point in points:  # Loop through each point
        for e in point.eccentricity:  # Each point has a list of [eccentricity, time]
            Eccentricities.append(e[0])  # Append eccentricity value
            Times.append(e[1])  # Append the corresponding time
    return Eccentricities, Times

#ECCENTRICITY
plotE, plotT = Eccentricities()

plotE = np.array(plotE)
plotT = np.array(plotT)

valid_indices = ~np.isnan(plotE)

plotE = plotE[valid_indices]
plotT = plotT[valid_indices]


plt.figure(figsize=(12, 6))
plt.ylim(0, 1.1)
plt.grid(False, which='both', linestyle='--', color='gray', alpha=0.5)
plt.title('Eccentricity vs Time')
plt.xlabel('Time')
plt.ylabel('Eccentricity')

plt.scatter(plotT, plotE, alpha = 1, color="darkorange", s=10)
plt.show()

print(f"Average Eccentricity: {sum(plotE)/len(plotE)}")
print(f"Total Orbits: {len(plotE)}")

'''
#COUNTS

def CountPointsOverTime():
    Times = []
    for point in points:
        for e in point.eccentricity:
            Times.append(e[1])  # Collect the times from eccentricity data

    # Define time bins (you can adjust the number of bins as needed)
    time_bins = np.linspace(0, totTime, 20)  # Time range: from 0 to totTime, with 20 bins

    # Count the number of points within each time bin
    counts_per_time_bin = []
    for i in range(len(time_bins) - 1):
        count_in_bin = sum(1 for t in Times if time_bins[i] <= t < time_bins[i + 1])
        counts_per_time_bin.append(count_in_bin)

    return time_bins[:-1], counts_per_time_bin  # Time bins and the corresponding point counts

time_bins, counts_per_time_bin = CountPointsOverTime()
plt.figure(figsize=(8, 6))

plt.bar(time_bins, counts_per_time_bin, width=np.diff(time_bins), edgecolor="black", alpha=0.7)

plt.title('Number of Points Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Points')
plt.show()
'''


    
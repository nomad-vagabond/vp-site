---
layout: post
title: Example 1. Building Parametric Spline Curves with SciPy
permalink: /examples/spline-approx-scipy/example1.html
root: ../../
---


~~~python
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

def getControlPoints(knots, k):
    n = len(knots) - 1 - k
    cx = np.zeros(n)
    for i in range(n):
        tsum = 0
        for j in range(1, k+1):
            tsum += knots[i+j]
        cx[i] = float(tsum)/k
    return cx

#####  CONTROL POINTS   #####

x = [6, -2, 4, 6, 8, 14, 6]
y = [-3, 2, 5, 0, 5, 2, -3]

xmin, xmax = min(x), max(x) 
ymin, ymax = min(y), max(y)

n = len(x)
plotpoints = 100

#####  PARAMETRIC SPLINE REPRESENTATION  #####

k = 3
knotspace = range(n)

# find knot vector 
knots = si.InterpolatedUnivariateSpline(knotspace,knotspace,k=k).get_knots()
knots_full = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))

# define tuples of knot vector, coefficient vector (control points coordinates) and spline degree
tckX = knots_full, x, k
tckY = knots_full, y, k

# construct spline functions
splineX = si.UnivariateSpline._from_tck(tckX)
splineY = si.UnivariateSpline._from_tck(tckY)

# evaluate spline points
tP = np.linspace(knotspace[0], knotspace[-1], plotpoints)
xP = splineX(tP)
yP = splineY(tP)

# define coordinates of spline control points in knot space
cp = getControlPoints(knots_full, k)

#####  PLOT  #####

offset_x = (xmax - xmin)*0.05
offset_y = (ymax - ymin)*0.1
knotpoints_y = [ymin-offset_y*0.5]*len(knots_full)
knotpoints_x = [xmin-offset_x*0.5]*len(knots_full)

fig = plt.figure()
ax = fig.add_subplot(224)
ax.grid(True)
plt.plot(x, cp, '-og')

plt.plot(xP, tP, 'r', lw=2)
plt.plot(knotpoints_x, knots_full, '>', ms=6, color='black')
plt.ylim([knotspace[0] - offset_y, knotspace[-1] + offset_y])
plt.xlim([xmin - offset_x, xmax + offset_x])
plt.ylabel('t', rotation=0,  labelpad=20, fontweight='bold', fontsize=14) # position=(1.0,0)
plt.xlabel('x', labelpad=10, fontweight='bold', fontsize=14) # position=(1.0,0)
ax.invert_yaxis()
plt.title('Spline function x(t)')

ax = fig.add_subplot(221)
ax.grid(True)
plt.plot(cp, y, '-og')
plt.plot(tP, yP, 'r', lw=2)
plt.plot(knots_full,knotpoints_y, '^', ms=6, color='black')
plt.xlim([knotspace[0] - offset_x, knotspace[-1] + offset_x])
plt.ylim([ymin - offset_y, ymax + offset_y])
plt.ylabel('y', labelpad=10, rotation=0, fontweight='bold', fontsize=14)
plt.xlabel('t', labelpad=20, fontweight='bold', fontsize=14)  #position=(0,0)
ax.invert_xaxis()
plt.title('Spline function y(t)')

ax = fig.add_subplot(222)
ax.grid(True)
plt.plot(x, y, '-og')
plt.plot(xP, yP, 'b', lw=2.5)
plt.xlim([xmin - offset_x, xmax + offset_x])
plt.ylim([ymin - offset_y, ymax + offset_y])
plt.title('Spline curve f(x(t), y(t))')

plt.show()
~~~
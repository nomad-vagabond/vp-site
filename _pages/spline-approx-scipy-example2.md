---
layout: post
title: Example 2. Spline Approximation with SciPy
permalink: /examples/spline-approx-scipy/example2.html
root: ../../
---

~~~python
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
from string import split, join

def readXY(filepath="./XYdata.txt"):
    with open(filepath, "r") as data:
        X = []
        Y = []
        for line in data:
            splitline = split(line)
            try:
                X.append(float(splitline[0]))
                Y.append(float(splitline[1]))
            except:
                pass
    return X, Y

def getControlPoints(knots, k):
    n = len(knots) - 1 - k
    cx = np.zeros(n)
    for i in range(n):
        tsum = 0
        for j in range(1,k + 1):
            tsum += knots[i+j]
        cx[i] = float(tsum)/k
    return cx

def plotSplineFunction(title="Spline Function", offset_x=0.5, offset_y=0.05):
    plt.plot(coeffs_x, coeffs_y, 'go-')
    plt.plot(x, y, 'ro')
    plt.plot(xP, yP, lw=2)
    plt.plot(knots, knotsy, 'gs') # ,markersize=10
    plt.axis([xmin-offset_x, xmax+offset_x, ymin-offset_y, ymax+offset_y])
    plt.grid(True)
    plt.title(title)
    plt.show()

def findLSQSplineRep():
    global knots, knots_full, coeffs_y, coeffs_x, yP, knotsy
    lsqspline = si.LSQUnivariateSpline(x, y, knots, k=k, w=w)
    knots = lsqspline.get_knots()
    coeffs_y = lsqspline.get_coeffs()
    yP = lsqspline(xP)
    knotsy = lsqspline(knots)
    knots_full = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))
    coeffs_x = getControlPoints(knots_full, k)

def findSmoothedSplineRep(s=0.005):
    global knots, knots_full, coeffs_y, coeffs_x, yP, knotsy
    spline = si.UnivariateSpline( x, y, k=k , s=s, w=w)
    knots = spline.get_knots()
    coeffs_y = spline.get_coeffs()
    yP = spline(xP)
    knotsy = spline(knots)
    knots_full = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))
    coeffs_x = getControlPoints(knots_full, k)

##### LOAD DATA POINTS #####

x, y = readXY("./data_charts/cy_f004.txt")

# print "x:", x
# print "y:", y

### set variables
nsample = 100
nknot = 3
k = 3
num_points = len(x)
xmin, xmax = min(x), max(x) 
ymin, ymax = min(y), max(y)
xP = np.linspace(x[0], x[-1], nsample)
# xP = np.linspace( x[0]-2, x[-1]+2, nsample )

plt.plot(x, y, 'ro')
plt.axis([xmin-0.5, xmax+0.5, ymin-0.05, ymax+0.05])
plt.grid(True)
plt.title("Data points")
plt.show()

### define weight vector to push further approximations stick to end points
wend = 3
w = [wend] + [1]*(num_points-2) + [wend]

##### FIND LSQ SPLINE REPRESENTATION  #####

### try with uniform knot vector

knot_offset = (xmax - xmin)/(nknot + 1)
knots = np.linspace(knot_offset, xmax-knot_offset, nknot)

findLSQSplineRep()
plotSplineFunction(title="LSQ spline function with uniform knot vector", offset_x=1.)

### manually select best fitting non-uniform knot vector

knots = [1.2, 1.85, 2.1, 2.4, 2.6] # !!
knots = [1.2, 1.85, 2.3, 2.6, 3.2] # !
knots = [1.2, 1.85, 2.0, 2.7, 3.2] # !!!
knots = [1.2, 1.85, 2.0, 2.8, 3.5] # !!!!

findLSQSplineRep()
plotSplineFunction(title="LSQ spline function with manual selected non-uniform knot vector", offset_x=1.)

#####  FIND SMOOTHED SPLINE REPRESENTATION   #####

findSmoothedSplineRep(s=0.1)
plotSplineFunction(title="Smoothed spline function s=0.1")
# print "s=0.1, knots:", knots
# findSmoothedSplineRep(s=0)
# plotSplineFunction(title = "Polynomial spline interpolation")
# print "s=0, knots:", knots
findSmoothedSplineRep()
plotSplineFunction(title="Smoothed spline function s=0.005")
# print "s=0.05, knots:", knots

#####   MANUALLY SELECT BEST_FITTING CONTROL POINTS   #####

### try to comment next two rows to look at the original smoothed spline curve after parametrization
coeffs_x = [x[0], 1.1, 1.7, 2.16, 3.1, 3.0, x[-1]]
coeffs_y = [y[0], 0.11, 0.17, 0.32, 0.27, 0.19, y[-1]]

#####   PARAMETRIC SPLINE REPRESENTATION   #####

### norm knot vector to number of points

num_knots = len(knots_full)
ka = (knots_full[-1] - knots_full[0])/(num_points)
knotsp = np.zeros(num_knots)
for i in range(num_knots):
 knotsp[i] = num_points-((knots_full[-1] - knots_full[i]))/ka
# print "knotsp:", knotsp

### find parametric spline representation based on control points coordinates (coeffs_x, coeffs_y)

tckX = knotsp, coeffs_x, k
tckY = knotsp, coeffs_y, k

splineX = si.UnivariateSpline._from_tck(tckX)
splineY = si.UnivariateSpline._from_tck(tckY)

coeffs_p = getControlPoints(knotsp, k)
tP = np.linspace(0, num_points, 100)
# tP = np.linspace(-20,  num_points+20, 100)
xP = splineX(tP)
yP = splineY(tP)

#####   PLOT PARAMETRIC SPLINE CURVE   #####

offset_x = (xmax-xmin)*0.05
offset_y = (ymax-ymin)*0.1
ct = 0.5
tadd = 0.5

# offset_x = (xmax-xmin)*0.2
# offset_y = (ymax-ymin)*0.2
# ct = 0.7
# tadd = 1.0

knotpoints_y = [ymin-offset_y*ct]*len(knotsp)
knotpoints_x = [xmin-offset_x*ct]*len(knotsp)

fig = plt.figure()

ax = fig.add_subplot(224)
ax.grid(True)
plt.plot(coeffs_x, coeffs_p, '-og')
plt.plot(xP, tP, 'r', lw=2)
plt.plot(knotpoints_x, knotsp, '>', ms=6, color='black')
plt.ylim([-tadd, num_points+tadd])
plt.xlim([xmin - offset_x, xmax + offset_x])
plt.ylabel('t', rotation=0,  labelpad=20, fontweight='bold', fontsize=14)
plt.xlabel('x', labelpad=5, fontweight='bold', fontsize=14)
ax.invert_yaxis()
plt.title('Spline function x(t)')

ax = fig.add_subplot(221)
ax.grid(True)
plt.plot(coeffs_p, coeffs_y, '-og')
plt.plot(tP, yP, 'r', lw=2)
plt.plot(knotsp,knotpoints_y, '^', ms=6, color='black')
plt.xlim([-tadd, num_points+tadd])
plt.ylim([ymin - offset_y, ymax + offset_y])
plt.ylabel('y', labelpad=20, rotation=0, fontweight='bold', fontsize=14)
plt.xlabel('t', labelpad=10, fontweight='bold', fontsize=14)
ax.invert_xaxis()
plt.title('Spline function y(t)')

ax = fig.add_subplot(222)
ax.grid(True)
plt.plot(x, y, 'ro')
plt.plot(coeffs_x, coeffs_y, '-og')
plt.plot(xP, yP, 'b', lw=2.5)
plt.xlim([xmin - offset_x, xmax + offset_x])
plt.ylim([ymin - offset_y, ymax + offset_y])
plt.title('Spline curve f(x(t), y(t))')

plt.show()
~~~
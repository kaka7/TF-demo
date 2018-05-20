#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#!python
from numpy import *
import pylab as p
#import matplotlib.axes3d as p3
import mpl_toolkits.mplot3d.axes3d as p3

# u and v are parametric variables.
# u is an array from 0 to 2*pi, with 100 elements
u=r_[0:2*pi:100j]
# v is an array from 0 to 2*pi, with 100 elements
v=r_[0:pi:100j]
# x, y, and z are the coordinates of the points for plotting
# each is arranged in a 100x100 array
x=10*outer(cos(u),sin(v))
y=10*outer(sin(u),sin(v))
z=10*outer(ones(size(u)),cos(v))

# fig=p.figure()
# ax = p3.Axes3D(fig)
# ax.plot_wireframe(x,y,z)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# p.show()

#!python
# this connects each of the points with lines
fig=p.figure()
ax = p3.Axes3D(fig)
# plot3D requires a 1D array for x, y, and z
# ravel() converts the 100x100 array into a 1x10000 array
ax.plot3D(ravel(x),ravel(y),ravel(z))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.add_axes(ax)
p.show()



#!python

delta = 0.025
x = range(-3.0, 3.0, delta)
y = range(-2.0, 2.0, delta)
X, Y = p.meshgrid(x, y)
Z1 = p.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = p.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)
fig=p.figure()
ax = p3.Axes3D(fig)
ax.contour3D(X,Y,Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
p.show()

#!python
# in mplt3D change:
# levels, colls = self.contourf(X, Y, Z, 20)
# to:
# C = self.contourf(X, Y, Z, *args, **kwargs)
# levels, colls = (C.levels, C.collections)
fig=p.figure()
ax = p3.Axes3D(fig)
ax.contourf3D(X,Y,Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.add_axes(ax)
p.show()
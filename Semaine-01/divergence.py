from numpy import pi, cos, sin
import numpy as np
import pylab as plt

x = np.linspace(-3,3,200)
y = np.linspace(-3,3,200)
X,Y = np.meshgrid(x,y)

U = np.sin(X)**2+Y
V = np.cos(X+Y**2)

dUdy, dUdx = np.gradient(U)
dVdy, dVdx = np.gradient(V)

div = dUdx*200./6. + dVdy*200./6. # to scale dx, dy

plt.pcolor(X,Y,div)
# plt.axes().set_aspect('equal')
plt.colorbar()

plt.streamplot(X,Y,U,V,arrowsize=2.,color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-3,3])
plt.ylim([-3,3])
# plt.axes().set_aspect('equal')

plt.show()
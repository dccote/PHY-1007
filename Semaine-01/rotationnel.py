from numpy import pi, cos, sin
import numpy as np
import pylab as plt

x = np.linspace(-3,3,200)
y = np.linspace(-3,3,200)
X,Y = np.meshgrid(x,y)

U = np.sin(X)**2+Y
V = np.cos(X+Y**2)

# 0 autour de (0.85, 2)
plt.quiver(X[::5,::5],Y[::5,::5],U[::5,::5],V[::5,::5],color='k',units='xy',scale=12, headwidth=10, headlength=15)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0.25,1.45])
plt.ylim([-2.6,-1.4])
plt.axes().set_aspect('equal')
plt.show()

# autour du max
plt.quiver(X[::5,::5],Y[::5,::5],U[::5,::5],V[::5,::5],color='k',units='xy',scale=12, headwidth=10, headlength=15)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([1,2])
plt.ylim([-1,0])
plt.axes().set_aspect('equal')
plt.show()

dUdy, dUdx = np.gradient(U)
dVdy, dVdx = np.gradient(V)

# dUdy = np.ones(dUdy.shape)
rot = dVdx*200./6. - dUdy*200./6. # to scale dx, dy

plt.pcolor(X,Y,rot)
plt.axes().set_aspect('equal')
plt.colorbar()

plt.streamplot(X,Y,U,V,arrowsize=2.,color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axes().set_aspect('equal')

plt.show()


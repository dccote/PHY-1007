from numpy import pi, cos, sin, sqrt
import numpy as np
import pylab as plt

from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-3,3,200)
y = np.linspace(-3,3,200)
X,Y = np.meshgrid(x,y)

# Champs scalaire
I = cos(Y+X)**2 + sin(X/3.)

# carte de couleur
plt.pcolor(X,Y,I)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.colorbar()
# plt.axes().set_aspect('equal')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, I, rstride=10, cstride=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.show()

# Champs vectoriel
U = np.sin(X/2 + 1)
V = np.cos(Y**2) + X

nb = 15
plt.quiver(X[::nb,::nb],Y[::nb,::nb],U[::nb,::nb],V[::nb,::nb],color='blue',units='xy',scale=4, headwidth=8, headlength=12)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-3,3])
plt.ylim([-3,3])
# plt.axes().set_aspect('equal')
# plt.show()

plt.streamplot(X,Y,U,V,arrowsize=2.,color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-3,3])
plt.ylim([-3,3])
# plt.axes().set_aspect('equal')

plt.show()


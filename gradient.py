from numpy import pi, cos, sin
import numpy as np
import pylab as plt

x = np.linspace(-pi,pi,200)
y = np.linspace(-pi,pi,200)
X,Y = np.meshgrid(x,y)

I = cos(Y)**2 + sin(X)**2
#I = Y**2 + 3

plt.pcolor(X,Y,I)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-pi,pi])
plt.ylim([-pi,pi])
plt.colorbar()
# plt.axes().set_aspect('equal')
# plt.show()

GY, GX = np.gradient(I)

plt.quiver(X[::15,::15],Y[::15,::15],GX[::15,::15],GY[::15,::15])
#plt.streamplot(X,Z,Eix,Eiz,color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-pi,pi])
plt.ylim([-pi,pi])
# plt.axes().set_aspect('equal')
plt.show()


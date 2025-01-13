from sympy.physics.vector import *
from sympy import sin, cos, sqrt

R = ReferenceFrame('R')

v = (sin(R[0])**2+R[1])*R.x + cos(R[0]+R[1]**2)*R.y
div_v = divergence(v,R)

# Region d'interet
xbounds = (0,1.)
ybounds = (-1.75,-0.75)

print('Integrale par segment sur le contour carre')
intval = 0
c1 = v.dot(-R.y)
i = (c1.integrate((R[0],xbounds[0],xbounds[1]))).subs({R[1]:ybounds[0]})
print('Segment du bas:',i)
intval += i
c2 = v.dot(R.x) 
i = (c2.integrate((R[1],ybounds[0],ybounds[1]))).subs({R[0]:xbounds[1]})
intval += i
print('Segment de droite:',i)
c3 = v.dot(R.y)
i = (c3.integrate((R[0],xbounds[0],xbounds[1]))).subs({R[1]:ybounds[1]})
intval += i
print('Segment du haut:',i)
c4 = v.dot(-R.x) 
i = (c4.integrate((R[1],ybounds[0],ybounds[1]))).subs({R[0]:xbounds[0]})
print('Segment de gauche:',i)
intval += i
print('Somme sur tout le contour:',intval,'\n')

# eval divergence en un point
#div_v.subs({R[0]:0,R[1]:0}).evalf()
print('Integrons la divergence sur toute la surface d\'interet:')
intgrl = div_v.integrate((R[0],xbounds[0],xbounds[1])).integrate((R[1],ybounds[0],ybounds[1]))
print(intgrl)

# pour figures:
from numpy import pi, cos, sin
import numpy as np
import pylab as plt

x = np.linspace(-3,3,200)
y = np.linspace(-3,3,200)
X,Y = np.meshgrid(x,y)

U = np.sin(X)**2+Y
V = np.cos(X+Y**2)

# vecteurs dans la region d'interet
plt.quiver(X[::3,::3],Y[::3,::3],U[::3,::3],V[::3,::3],color='k',units='xy',scale=12, headwidth=10, headlength=15)
plt.xlabel('')
plt.ylabel('')
plt.xlim(xbounds)
plt.ylim(ybounds)
# plt.axes().set_aspect('equal')
plt.gca().set_xticks([]) 
plt.gca().set_yticks([]) 
plt.title('Champ de vecteur dans la region d\'interet')
plt.show()

dUdy, dUdx = np.gradient(U)
dVdy, dVdx = np.gradient(V)
div = dUdx*200./6. + dVdy*200./6. # divise par dx et dy

plt.pcolor(X,Y,div)
# plt.axes().set_aspect('equal')
plt.colorbar()
plt.xlabel('')
plt.ylabel('')
plt.xlim(xbounds)
plt.ylim(ybounds)
# plt.axes().set_aspect('equal')
plt.gca().set_xticks([]) 
plt.gca().set_yticks([]) 
plt.title('Divergence dans la region d\'interet')
plt.show()

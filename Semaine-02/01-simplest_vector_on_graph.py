import matplotlib.pyplot as plt 

""" La fonction quiver (i.e. en anglais quiver = carquois pour tenir des flèches)
est très simple: on passe les coordonnées x,y et les longueurs u,v des vecteurs que
l'on veut tracer. Chaque valeur est dans une list séparée. Ici, on aura donc deux 
vecteurs:

Au point (0,0), un vecteur avec les composantes (2,1) dans les unités du système de
coordonnées.
Au point (-2,1), un vecteur avec les composantes (3,-2) dans les unités du système de
coordonnées.

"""

X = [0,-2]
Y = [0, 1] 

U = [4, 7] 
V = [2,-3] 

plt.quiver(X, Y, U, V) 
plt.xlim(-5, 5) 
plt.ylim(-5, 5) 

plt.grid() 
plt.show() 

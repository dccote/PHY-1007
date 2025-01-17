import matplotlib.pyplot as plt 

""" La fonction quiver (i.e. en anglais quiver = carquois pour tenir des flèches)
est très simple: on passe les coordonnées x,y et les longueurs u,v des vecteurs que
l'on veut tracer. Chaque valeur est dans une list séparée. 

Ici, 9 vecteurs aux positions (-1,-1), ensuite (0,-1), (1,-1), ... (0,1) et (1,1)
avec des longueurs de 2, 3 et 4 sont ajoutés dans le graphique.

"""


X = [-1, 0, 1,-1, 0, 1,-1, 0, 1]
Y = [-1,-1,-1, 0, 0, 0, 1, 1, 1] 

U = [ 2, 2, 2, 3, 3, 3, 4, 4, 4] 
V = [ 0, 0, 0, 0, 0, 0, 0, 0, 0] 

plt.quiver(X, Y, U, V) 
plt.xlim(-2, 2) 
plt.ylim(-2, 2) 

plt.grid() 
plt.show() 

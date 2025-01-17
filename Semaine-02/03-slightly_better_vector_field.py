import matplotlib.pyplot as plt 
import numpy as np

""" La fonction quiver (i.e. en anglais quiver = carquois pour tenir des flèches)
est très simple: on passe les coordonnées x,y et les longueurs u,v des vecteurs que
l'on veut tracer. Chaque valeur est dans une list séparée. 

Pour facilement construire une liste de coordonnées sur un grille, on peut
utiliser les fonctions linspace et meshgrid de numpy: on lui donne nos
valeurs en X (seulement) et en Y (seulement) et ensuite la fonction mesh les
combinera pour compléter la grille.

"""

x = np.linspace(-1,1,9) # 9 valeurs différentes entre -1 et 1
y = np.linspace(-1,1,9) # 9 valeurs différentes entre -1 et 1
X,Y = np.meshgrid(x,y)  # Les 81 combinaisons possibles pour former la grille


U = [ 2, 2, 2, 3, 3, 3, 4, 4, 4] * 9 # Je sauve du temps et je repete 9 fois la meme liste
V = [ 0 ] * 81

plt.quiver(X, Y, U, V) 
plt.xlim(-2, 2) 
plt.ylim(-2, 2) 

plt.grid() 
plt.show() 

import matplotlib.pyplot as plt 
import numpy as np

""" La fonction quiver (i.e. en anglais quiver = carquois pour tenir des flèches)
est très simple: on passe les coordonnées x,y et les longueurs u,v des vecteurs que
l'on veut tracer. Chaque valeur est dans une list séparée. 

Pour facilement construire une liste de coordonnées sur un grille, on peut utiliser
la fonction mesh de numpy: on lui donne nos valeurs en X (seulement) et en Y (seulement)
et ensuite la fonction mesh les combinera pour compléter la grille.

Encore mieux, on peut utiliser X et Y (qui sont des arrays numpy) comme des variables.
Ainsi, en ecrivant "sin(X)" on aura le resultat de la fonction pour chaque valeur de la liste.
Les listes U et V auront automatiquement la même longueurs que X et Y, et les éléments
pour chaque indice seront des éléments correspondant.

"""

x = np.linspace(-1,1,9) # 9 valeurs différentes entre -1 et 1
y = np.linspace(-1,1,9) # 9 valeurs différentes entre -1 et 1
X,Y = np.meshgrid(x,y)  # Les 81 combinaisons possibles pour former la grille


U = np.sin(2*X)
V = np.cos(2*Y)

plt.quiver(X, Y, U, V) # On peut ajuster les échelles (scale et units) : voir 10-testing_vector_fields.py
plt.xlim(-2, 2) 
plt.ylim(-2, 2) 

plt.grid() 
plt.show() 

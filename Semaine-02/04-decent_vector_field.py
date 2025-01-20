import matplotlib.pyplot as plt
import numpy as np

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
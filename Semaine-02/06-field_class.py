import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize


"""

Nous avons le code suivant pour faire un graphique d'un champ vectoriel avec des paramètres raisonnables:

```
N = 19

x = np.linspace(-10,10,N)
y = np.linspace(-10,10,N)
X,Y = np.meshgrid(x,y)

U = np.sin(X/2)
V = np.cos(Y/2)

lengths = np.sqrt(U*U+V*V)
plt.tick_params(direction="in")
plt.quiver(X, Y, U, V, lengths, units='xy', scale=1, width=0.15)
plt.show() 
```

Il suffirait de le recopier a chaque fois que l'on veut l'utiliser.  On
pourrait aussi le mettre dans une fonction, qu'on mettrait dans un fichier,
et qui definirait une fonction qu'on pourrait utiliser a l'appelant avec les
paramètres.  Ce serait une fonction affreuse:

def plot_vector_field(x_min, x_max, y_min, y_max, N, X,Y,U,V, scaling, colors):
	# blabla


et meme si je l'utilisais, je manque de flexibilité et je dois quand même
générer mon champ avant (U et V). Il y a une bien meilleure facon, c'est en
faisant des classes pour gérer de façon efficace la complexité du problème:
clairement, nous voulons créer et manipuler un champ vectoriel, c'est clairement 
le concept le plus important et central de notre probleme.

Nous allons donc créer une classe (VectorField) qui contiendra toutes les données
pour décrire le champs, mais aussi des méthodes pour nous aider à visualiser, manipuler 
le champ. Ainsi, nous pourrions utiliser notre code simplement comme suit:

```
from vectorfield import VectorField


e_field = VectorField(X, Y, U, V)
e_field.display()

```

Pour commencer, je fais le strict minimum pour reproduire ce que nous avions auparavant:

1. Afficher de -10 a 10
2. Mettre 19 points par défaut
3. Mettre un champ par défaut si aucun n'est fourni

"""

class VectorField2D:
	def __init__(self, X=None, Y=None, U=None, V=None):
		if X is None or Y is None:
			x = np.linspace(-10,10,19)
			y = np.linspace(-10,10,19)
			X,Y = np.meshgrid(x,y)

		self.X = X
		self.Y = Y

		if U is None or V is None:
			U = np.sin(self.X/2)
			V = np.cos(self.Y/2)

		self.U = U
		self.V = V

	def display(self):
		plt.tick_params(direction="in")

		lengths = np.sqrt(self.U*self.U+self.V*self.V)
		plt.quiver(self.X, self.Y, self.U, self.V, lengths, units='xy', scale=1, width=0.15)
		plt.show() 


field = VectorField2D()
field.display() # Le défaut est un champ constant

# Je peux changer le champ en assignant directement les variables U et V. C'est moche, mais ca fonctionne:
field.U = field.X/10
field.V = field.Y*field.Y/100
field.display()

"""
Conclusion: j'ai un peu avancé, car je peux maintenant utiliser une classe qui englobe tous les détails de mon champ
et me donne une méthode display() pour l'afficher.  Je peux mettre des paramètres par défaut et changer les variables
a la main par la suite, même changer le champ.

C'est un peu mieux, mais il y a encore plusieurs amélioration possibles:

1. Il faudrait trouver une facon de rapidement choisir des exemples de champs (charge simple, dipole, sphère, etc...)
2. Il faudrait raffiner le contrôle du champ
3. Il faudrait permettre l'ajustement des détails de l'affichage
4. Il faudrait valider les données pour éviter des variables X,Y et U,V incompatibles
5. Et encore plus...


"""
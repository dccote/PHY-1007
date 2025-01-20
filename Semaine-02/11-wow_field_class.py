import matplotlib.pyplot as plt 
import numpy as np


"""
Je trouve que les couleurs ne sont pas terribles: puisqu'il y a souvent des valeurs
tres grandes (i.e. infinies) a l'origine des charges, toutes les petites
valeurs se retrouvent de la meme couleur.  Je vais plutot limiter les valeurs
entre le 10e et 90e percentile au lieu de normaliser sur la valeur maximale.

De plus, on voit que maintenant j'appelle les fonctions comme suit:

	U, V = field.create_single_charge_field_components(xo=3, yo=0, q=1)
	field.add_field_components(U, V)

donc je crois qu'il est plus simple de créer une nouvelle fonction
qui fera tout d'un seul appel de fonction:

field.add_single_charge(xo=3, yo=0, q=1)

Je garde quand même les anciennes fonctions, car je ne suis pas certain
si je vais le réutiliser. Cependant, j'enleve le champ par defaut en sinus et cosinus
que j'avais utilisé depuis le debut.

"""

class SurfaceDomain:
	def __init__(self, size=20, N=19, X=None, Y=None):
		"""
		Le domain par défaut va de -10 a 10 en X et Y, et discrétise
		avec N=19 points. L'utilisateur peut quand même fournir ses np.array
		X et Y, déjà calculés d'avance.
		"""
		if size is not None:
			X, Y = self.create_square_meshgrid(size, N)
		elif X is None or Y is None:
			X, Y = self.create_square_meshgrid()

		if len(X) != len(Y):
			raise ValueError("Les composantes X et Y doivent avoir le meme nombre d'éléments")

		self._X = X # Lorsqu'on utilise _ devant une variable, c'est une convention de ne pas l'appeler directement
		self._Y = Y # et d'utiliser les @property accessors

	def xy_mesh(self, xo=0, yo=0):
		"""
		Les np.arrays X,Y du meshgrid, mais relatif à l'origine (xo, yo).
		Ceci permet d'utiliser directement les valeurs pour le calcul du 
		champ d'une charge unique.

		Par défaut, l'origine est à (0,0)
		"""
		return self._X-xo, self._Y-yo

	def rphi_mesh(self, xo=0, yo=0):
		"""
		Les np.arrays R,PHI du meshgrid, mais relatif à l'origine (xo, yo).
		Ceci permet d'utiliser directement les valeurs pour le calcul du 
		champ d'une charge unique.

		Par défaut, l'origine est à (0,0)
		"""
		X,Y = self.xy_mesh(xo, yo)
		return np.sqrt(X*X+Y*Y), np.atan2(Y, X)

	def create_square_meshgrid(self, size=20, N=19):
		"""
		Fonction pour créer un domaine XY rapidement
		"""
		x = np.linspace(-size/2,size/2, N)
		y = np.linspace(-size/2,size/2, N)
		return np.meshgrid(x,y)

class VectorField2D:
	def __init__(self, surface=None, U=None, V=None):
		"""
		On accepte la surface ou on en crée une par défaut. L'utilisteur peut
		fournir les composantes d'un champ U, V mais sinon, le champ est
		initialisé à 0,0 partout.
		"""
		if surface is None:
			surface = SurfaceDomain()

		self.domain = surface

		if U is None or V is None:
			U, V = self.create_null_field_components()

		self.U = U
		self.V = V

		self.quiver_axes = None

		self.validate_arrays() # Avant d'aller plus loin, nous voulons un champ valide


	@property
	def field_magnitude(self):
		return np.sqrt(self.U*self.U+self.V*self.V)

	def create_null_field_components(self):		
		X,Y = self.domain.xy_mesh()
		return X*0, X*0 # C'est un truc pour avoir rapidement une liste de la meme longueur avec des zeros

	def create_single_charge_field_components(self, xo=0, yo=0, q=1):

		# On pourra se retrouver, parfois, avec R==0 (a l'origin), et donc 1/(R*R) 
		# sera infini. On veut éviter les infinités et les discontinuités.
		# Pour l'instant, pour simplifer, je vais simplement ajouter un tout
		# petit 0.01 a R*R pour eviter que cela donne une division par zero.
		# C'est affreux, mais ca évite les problèmes.
		R, PHI = self.domain.rphi_mesh(xo, yo)

		return q*np.cos(PHI)/(R*R+0.01), q*np.sin(PHI)/(R*R+0.01)

	def add_field_components(self, U, V):
		self.U += U
		self.V += V
		self.validate_arrays()

		self.quiver_axes = None # Nous avons changé le champ, la figure n'est plus valide

	def add_single_charge(self, xo=0, yo=0, q=1):
		U, V = self.create_single_charge_field_components(xo, yo, q)
		self.add_field_components(U, V)

	def validate_arrays(self):
		if self.U is None or self.V is None:
			raise ValueError("Les composantes U et V du champ vectoriel ne sont pas assignées en tout point X et Y")

		if len(self.U) != len(self.V):
			raise ValueError("Les composantes U et V doivent avoir le meme nombre d'éléments")

	def display(self, use_color=True, title=None):
		self.validate_arrays()

		if self.quiver_axes is None:
			self.quiver_axes = plt.subplot(1,1,1)
			self.quiver_axes.tick_params(direction="in")

		self.quiver_axes.cla()

		X,Y = self.domain.xy_mesh()

		if use_color:
			"""
			Au lieu de prendre la longueur de la fleche pour représenter
			la force du champ, je garde les fleches de la meme longueur
			et je les colore en fonction de la force du champ.
			"""

			lengths = self.field_magnitude

			# Les couleurs sont biaisées car il y a souvent des valeurs tres grandes.
			# PLutot que de normaliser sur la plus grande valeurs, je limite 
			# les valeurs entre les percentiles 10-90 et je normalise la longueur des fleches. 
			# Ca fait plus beau.
			percentile_10th = np.percentile(lengths, 10)
			percentile_90th = np.percentile(lengths, 90)
			colors = np.clip(lengths, a_min=percentile_10th, a_max=percentile_90th)

			# Et finalement, j'ai compris que les unités du champ sont plus simple 
			# lorsqu'on prend relatif a la grandeur du graphique: la largeur
			# de la fleche sera aussi mieux adaptée independamment des unités.
			self.quiver_axes.quiver(X, Y, self.U/lengths, self.V/lengths, colors)
		else:
			self.quiver_axes.quiver(X, Y, self.U, self.V)

		plt.title(title)
		plt.show()
		self.quiver_axes = None


if __name__ == "__main__": # C'est la façon rigoureuse d'ajouter du code après une classe
	single_charge_field = VectorField2D() # Le domaine par défaut est -10 a 10 avec 19 points par dimension
	single_charge_field.add_single_charge(xo=0, yo=0, q=1)
	single_charge_field.display(use_color=True, title="Une seule charge")

	two_charges_field = VectorField2D()
	two_charges_field.add_single_charge(xo=5, yo=0, q=1)
	two_charges_field.add_single_charge(xo=-5, yo=0, q=1)
	two_charges_field.display(use_color=True, title="Deux charges positives")

	dipole_field = VectorField2D()
	dipole_field.add_single_charge(xo=5, yo=0, q=1)
	dipole_field.add_single_charge(xo=-5, yo=0, q=-1)
	dipole_field.display(use_color=True, title="Dipole")

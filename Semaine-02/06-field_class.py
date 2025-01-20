import matplotlib.pyplot as plt
import numpy as np

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
		plt.quiver(self.X, self.Y, self.U, self.V, lengths)
		plt.show()


field = VectorField2D()
field.display() # Le défaut est un champ constant

# Je peux changer le champ en assignant directement les variables U et V. C'est moche, mais ca fonctionne:
field.U = field.X/10
field.V = field.Y*field.Y/100
field.display()
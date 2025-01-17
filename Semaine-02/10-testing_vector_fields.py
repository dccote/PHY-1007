import unittest
import inspect
import matplotlib.pyplot as plt 

class QuiverTestCase(unittest.TestCase):

    def get_function_name(self, level=1):
        """
        Fonction qui permet d'obtenir le nom de la fonction qui a été appelé
        pour nomer le graphique avec le même nom.
        """
        return f"{inspect.stack()[level][3]}()"

    def test_01_basic_2d_quiver(self):
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

        U = [2, 3] 
        V = [1,-2] 

        plt.quiver(X, Y, U, V, color='b', units='xy', scale=1) 
        plt.xlim(-5, 5) 
        plt.ylim(-5, 5) 

        plt.grid() 
        plt.title(self.get_function_name())
        plt.show() 


    def test_02_basic_2d_quiver_different_units(self):
        """ 
        Les composantes du vecteur peuvent être données dans d'autres unités.
        Les possibilités sont: {'width', 'height', 'dots', 'inches', 'x', 'y', 'xy'}

        """

        X = [0,-2]
        Y = [-2, 1] 

        U = [0, 0.5] # en unités de largeur de graphique
        V = [0.5, 0] # en unités de largeur ("width") de graphique

        plt.quiver(X, Y, U, V, color='b', units='width', scale=1) # Changez units
        plt.xlim(-5, 5) 
        plt.ylim(-5, 5) 

        plt.grid() 
        plt.title(self.get_function_name())
        plt.show() 

    def test_03_basic_2d_quiver_different_scale(self):
        """ 
        On peut aussi ajouter un facteur de division : scale=2 va diviser la longueur
        par deux.
        """

        X = [ 0,-2]
        Y = [-2, 1] 

        U = [3, 4]
        V = [2, 5]

        plt.quiver(X, Y, U, V, color='b', units='xy', scale=2) 
        plt.xlim(-5, 5) 
        plt.ylim(-5, 5) 

        plt.grid() 
        plt.title(self.get_function_name())
        plt.show() 

    def test_04_field_2d_quiver(self):
        """ 
        On peut construire 
        """

        X = [ 0,-2]
        Y = [-2, 1] 

        U = [3, 4]
        V = [2, 5]

        plt.quiver(X, Y, U, V, color='b', units='xy', scale=2) 
        plt.xlim(-5, 5) 
        plt.ylim(-5, 5) 

        plt.grid() 
        plt.title(self.get_function_name())
        plt.show() 

if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import zoom

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
except:
    print(
        "OpenCL not available. On Linux: sudo apt-get install python-pyopencl seems to work (Ubuntu ARM64 macOS)."
    )
    cl = None
    cl_array = None

"""
Qu'est-ce que ce fichier?

Ceci est ce qu'on appelle un fichier de tests (ou un fichier de tests unitaires, Unit Tests).

Qu'est-ce que cela me permet de faire?

Les tests unitaires permettent de valider notre compréhension de certaines
fonctions (en validant les valeurs retournées) ou en testant des fonctions
que nous mettons en place. Il s'agit donc d'une série de tests qui représente
l'évolution de ma compréhension des arrays numpy des slices, de la méthode de
la relaxation pour solutionner l'équation de Laplace, et meme du code GPU a
la fin.

Vous n'avez qu'a rouler le code `python test_laplace.py`. Si vous n'avez pas
PyOpenCL, les tests de GPU ne fonctionneront simplement pas.

`pip install pyopencl`

"""

TOLERANCE = 1e-5


class ArrayManipulationTestCase(unittest.TestCase):
    def test01_init(self):
        """
        On confirme que tout est bien mis en place.
        """
        self.assertTrue(True)

    def test02_slice1D(self):
        """
        Comment fonctionne les slices? Si je comprends bien:

        [0:3] veut dire 0,1,2
        [1:-1] veut dire du deuxieme à l'avant dernier
        [0:-1] veut dire du premier à l'avant dernier
        """
        a = np.array([1, 2, 3])
        self.assertTrue((a[0:3] == [1, 2, 3]).all())
        self.assertTrue((a[1:-1] == [2]).all())
        self.assertTrue((a[:-1] == [1, 2]).all())

    def test03_slice2D(self):
        """
        Je continue mes tests pour bien comprendre les slices, maintenant
        je fais des slices en 2D. Le premier indice correspond a la grande
        liste, le deuxieme indice aux petites listes.

        """
        a_py = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a = np.array(a_py)
        self.assertEqual(a.shape, (3, 3))

        self.assertTrue((a[0:3, :] == a_py).all())
        self.assertTrue((a[1:-1, :] == [[4, 5, 6]]).all())
        self.assertTrue((a[:-1, :] == [[1, 2, 3], [4, 5, 6]]).all())

        self.assertTrue((a[:, :-1] == [[1, 2], [4, 5], [7, 8]]).all())

    def test04_slice2D_5x5(self):
        """
        Je replique maintenant ce que Louis Archambault a montré dans le cours
        pour faire la moyenne des 4 voisins les plus proches, on peut refaire
        4 matrices décalées.

        Je confirme que je comprends bien ce qui se passe.
        """

        a_py = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]
        a = np.array(a_py)
        self.assertEqual(a.shape, (5, 5))

        # 3x3, indice 1=top, indice 2=center
        self.assertTrue((a[:-2, 1:-1] == [[2, 3, 4], [7, 8, 9], [12, 13, 14]]).all())
        # 3x3, 1=center, 2=center
        self.assertTrue(
            (a[1:-1, 1:-1] == [[7, 8, 9], [12, 13, 14], [17, 18, 19]]).all()
        )
        # 3x3, 1=bottom, 2=center
        self.assertTrue(
            (a[2:, 1:-1] == [[12, 13, 14], [17, 18, 19], [22, 23, 24]]).all()
        )

        # 3x3, 1=center, 2=top
        self.assertTrue((a[1:-1, :-2] == [[6, 7, 8], [11, 12, 13], [16, 17, 18]]).all())
        # 3x3, 1=center, 2=center
        self.assertTrue(
            (a[1:-1, 1:-1] == [[7, 8, 9], [12, 13, 14], [17, 18, 19]]).all()
        )
        # 3x3, 1=center, 2=bottom
        self.assertTrue((a[1:-1, 2:] == [[8, 9, 10], [13, 14, 15], [18, 19, 20]]).all())

    def test05_mean1D(self):
        """
        Maintenant, j'utilise les slices pour faire des moyennes:

        j'additionne deux listes décalées (par la gauche de 1, par la droite de 1)
        et je confirme que j'obtiens ce que je suis supposé obtenir.
        """
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertTrue((a[:-2] == [1, 2, 3, 4, 5, 6, 7]).all())
        self.assertTrue((a[2:] == [3, 4, 5, 6, 7, 8, 9]).all())

        self.assertTrue((a[:-2] + a[2:] == [4, 6, 8, 10, 12, 14, 16]).all())
        self.assertTrue(((a[:-2] + a[2:]) / 2 == [2, 3, 4, 5, 6, 7, 8]).all())

    def test06_mean2D(self):
        """
        Je refais la même chose en 2D: je confirme que chaque sous-matrices
        que j'ai extraite est vraiment 3x3, et que la moyenne est calculée correctement.
        Ceci est en fait l'étape importante pour la solution de l'équation de Laplace
        par la méthode de la relaxation.

        La matrice 1,2,3,4,5... etc... a une particularité intéressante:
        la valeur est la moyenne de ses voisins.  Je peux donc verifier que tout
        fonctionne avec les valeurs de la matrices.

        """
        a_py = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]
        a = np.array(a_py)
        self.assertEqual(a.shape, (5, 5))
        self.assertEqual(a[2:, 1:-1].shape, (3, 3))
        self.assertEqual(a[:-2, 1:-1].shape, (3, 3))
        self.assertEqual(a[1:-1, :-2].shape, (3, 3))
        self.assertEqual(a[1:-1, 2:].shape, (3, 3))
        self.assertTrue(
            (
                a[1:-1, 1:-1]
                == (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]) / 4
            ).all()
        )

    def test07_laplace_convergence(self):
        """
        Je fais un tout petit test de la méthode de la relaxation pour vérifier si tout
        fonctionne. J'ai un carré avec 10V "en haut" et 0V sur les autres côtés,
        et je veux le potentiel ailleurs. Il n'y a que 5x5 elements.

        Ca devrait converge en moins de 2 secondes, sinon il y a un probleme.

        """
        a_py = [
            [10, 10, 10, 10, 10],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        a = np.array(a_py, dtype=np.float32)

        start_time = time.time()
        end_time = start_time + 2
        error = float("+inf")
        while error is None or error > TOLERANCE:
            before_iteration = a.copy()
            a[1:-1, 1:-1] = (
                a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
            ) / 4
            error = np.std(a - before_iteration)

            self.assertTrue(time.time() < end_time, "Did not converge fast enough")

    def test08_laplace_general(self):
        """
        Enfin, un vrai test: une matrice de 30x30 avec les mêmes conditions aux limites.
        Je montre le graphique des iterations a mesure.
        """
        a = np.zeros((30, 30), dtype=np.float32)
        a[0, :] = 10

        error = None
        iteration = 0
        while error is None or error > TOLERANCE:
            before_iteration = a.copy()
            a[1:-1, 1:-1] = (
                a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
            ) / 4
            error = np.std(a - before_iteration)
            if iteration % 100 == 0:
                plt.title(f"{self._testMethodName} #{iteration} err:{error:.7f}")
                plt.imshow(a)
                plt.pause(0.1)
            iteration += 1

    def solve_laplace2D_by_relaxation(
        self, v, initial_condition=None, tolerance=TOLERANCE
    ):
        """Ceci n'est pas un test: j'ecris une fonction qui me permettra de solutionner
        pour n'importe quelle matrice et n'importer quelle conditions aux limites.

        initial_condition est une fonction qui place les valeurs fixes dans la matrices.

        """

        error = None
        while error is None or error > tolerance:
            before_iteration = v.copy()
            v[1:-1, 1:-1] = (
                v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:]
            ) / 4
            if initial_condition is not None:
                initial_condition(v)

            error = np.std(v - before_iteration)

        return v

    def test09_define_slices(self):
        """
        Ceci est un petit test pour voir si je peux definir des slices au lieu d'avoir
        a les ecrire directement entre les []

        En fait, la classe "slice" de Python est exactement cela: elle represente la slice
        donc je peux definir une variable au lieu de "hardcoder" les valeurs
        """
        a = [1, 2, 3, 4, 5, 6]
        self.assertEqual(a[1:3], [2, 3])
        self.assertEqual(a[1:3], a[slice(1, 3)])
        self.assertEqual(a[1:-1], [2, 3, 4, 5])

    def set_initial_condition(self, v):
        """
        Ceci est un exemple de conditions initiales en 2D: le haut de l'image est a 10V
        et au centre, il y a un "pic" a 10V.

        Je retourne la matrice, mais ce n'est pas necesaire: elle est modifiée directement en place.
        """
        v[0, :] = 10
        # v[: , 0] = 0
        # v[-1, :] = 0
        # v[: ,-1] = 0
        # w0, w1 = v.shape

        # for i in range(w0//2):
        #   v[w0//4+i, w1//2] = 10

        return v

    def set_initial_condition3D(self, v):
        """
        Ceci est un exemple de conditions initiales en 3D: le haut de l'image est a 10V
        et au centre, il y a un "pic" a 10V.

        Je retourne la matrice, mais ce n'est pas necesaire: elle est modifiée directement en place.

        """
        v[0, :, :] = 10
        w0, w1, w2 = v.shape
        for i in range(5):
            v[w0 // 2, w1 // 2 + i, w2 // 2] = 10
            v[w0 // 2, w1 // 2 - i, w2 // 2] = 10

        return v

    def test10_laplace_initial_condition_fct(self):
        """
        test de la nouvelle fonction generale en 2D
        """
        v = np.zeros((240, 240), dtype=np.float32)

        start_time = time.time()
        self.set_initial_condition(v)
        self.solve_laplace2D_by_relaxation(
            v, initial_condition=self.set_initial_condition
        )

        print(f"\nCPU, 2D single grid {v.shape}: {time.time()-start_time:.3f}")

        plt.title(f"{self._testMethodName}")
        plt.imshow(v)
        plt.pause(0.1)

    def solve_laplace3D_by_relaxation(self, v, initial_condition=None):
        """
        On n'est pas peureux: on essaie en 3D. C'est la meme chose qu'en 2D mais maintenant,
        nous avons 6 voisins pour faire la moyenne.

        J'utilise les variables de slices pour simplifier le code.
        """
        error = None

        left = slice(0, -2)  # [0:-2]
        center = slice(1, -1)  # [1:-1]
        right = slice(2, None)  # [2:  ]

        start_time = time.time()
        iteration = 0
        while error is None or error > TOLERANCE:
            if iteration % 20 == 0:
                before_iteration = v.copy()
            v[center, center, center] = (
                v[left, center, center]
                + v[center, left, center]
                + v[center, center, left]
                + v[right, center, center]
                + v[center, right, center]
                + v[center, center, right]
            ) / 6
            if initial_condition is not None:
                initial_condition(v)

            if iteration % 20 == 0:
                error = np.std(v - before_iteration)

            iteration += 1

        return v

    def test11_laplace3D_initial_condition_fct(self):
        """
        On teste notre fonction qui solutionne en 3D l'equation de Laplace.
        C'est quand meme assez lent, mais ca fonctionne (30 secondes sur mon ordinateur)
        """

        v = np.zeros((50, 50, 50), dtype=np.float32)

        start_time = time.time()

        self.set_initial_condition3D(v)
        self.solve_laplace3D_by_relaxation(
            v, initial_condition=self.set_initial_condition3D
        )

        print(f"\nCPU, 3D single grid {v.shape}: {time.time()-start_time:.3f}")

        for i in range(v.shape[0], 10):
            plt.title(f"{self._testMethodName}: 3D layer={i}")
            plt.imshow(v[i, :, :])
            plt.pause(0.1)

    def test12_laplace_with_finer_and_finer_grid(self):
        """
        Ca fonctionne, mais la majorité des calculs sont asse3z rapides, mais
        si on veut une tres bonne résolution, alors ils deviennent un peu
        lent.  Je comprends par mes lectures que c'est souvent parce qu'on
        commence avec une matrice qui est tres loin de la vraie réponse.
        Il est donc utile de faire des solutions grossière pour ensuite
        rafiner la resolution a partir de la solution précédente. Je tente
        de le faire ici. A chaque fois, j'augmente la resolution par 2
        avec une interpolation linéaire.

        La solution (320,320) sans ce raffinement peut prendre jusqu'à 30
        secondes, mais seulement 20 secondes si on itere graduellement en
        raffinant la resolution a chaque fois.

        """
        v = np.zeros((32, 32), dtype=np.float32)

        start_time = time.time()
        with plt.ioff():
            for i in range(2):
                v = zoom(v, 2, order=0)
                self.set_initial_condition(v)
                self.solve_laplace2D_by_relaxation(
                    v, initial_condition=self.set_initial_condition
                )
                plt.title(f"{self._testMethodName} : # {i} {v.shape}")
                plt.imshow(v)
                plt.pause(0.1)

        print(f"\nCPU, with grid refinement {v.shape}: {time.time()-start_time:.3f}")


@unittest.skipIf(cl is None, "PyOpenCL is not installed.")
class OpenCLArrayTestCase(unittest.TestCase):
    def test01_2Dopencl(self):
        """
        Les calculs de Laplace par la relaxation sont faciles à mettre en place sur GPU
        car on refait le même calcul à répétition pour chaque element de la matrice.

        Je n'ai pas verifie la validite du resultat: je devrais faire le calcul en Python pur,
        ensuite le refaire sur GPU et montrer que les valeurs sont les memes, mais ce n'est pas encore fait.

        Je ne peux pas donner tous les détaisl des calculs GPU ici, mais voici
        un exemple en utilisanbt PyOpenCL qui fonctionne tres bien sur macOS
        et qui peut fonctionner sur Windows et Linux.

        J'importe les modules ici car je veux que ce fichier de test fonctionne
        même si les modules ne sont pas installés sur l'ordinateur des étudiants.
        """

        # Define a 2D array (float32)
        h, w = 128, 128
        size = h * w
        host_array = np.zeros(shape=(h, w), dtype=np.float32)
        host_array[0, :] = 10  # Initial conditions

        # Create OpenCL buffers: use cl.Array to benefit from operator overloading
        d_input = cl_array.to_device(self.queue, host_array)  # Copy data to GPU
        d_output = cl_array.empty_like(d_input)  # Create an empty GPU array

        # Set up the execution parameters
        global_size = (w, h)  # Matches the 2D array size

        start_time = time.time()
        stds = []
        plt.clf()
        for i in range(5000):
            # The calculation is sent to d_output, which I then use as the input for another iteration
            # This way, d_input becomes the output and I do not have to create an array each time.  This is very efficient.

            self.program.laplace(
                self.queue, global_size, None, d_input.data, d_output.data, np.int32(w)
            )
            self.program.laplace(
                self.queue, global_size, None, d_output.data, d_input.data, np.int32(w)
            )

            if i % 500 == 0:
                d_diff = d_output - d_input

                mean_val = (
                    cl_array.sum(d_diff).get() / size
                )  # Mean (transfers single float)
                d_diff_sq = (d_diff - mean_val) ** 2  # Element-wise (x - mean)^2
                variance_val = (
                    cl_array.sum(d_diff_sq).get() / size
                )  # Variance (transfers single float)
                max_val = cl_array.max(d_diff).get()
                min_val = cl_array.min(d_diff).get()
                std_val = np.sqrt(variance_val)  # Standard deviation (final sqrt)
                if std_val < TOLERANCE:
                    break
                stds.append(std_val)

        print(
            f"\nOpenCL, 2D single grid {host_array.shape}: {time.time() - start_time:.3f}"
        )

        plt.clf()
        plt.title(f"{self._testMethodName}: Error vs iteration")
        plt.plot(stds)
        plt.pause(0.1)

    def test02_3Dopencl(self):
        """
        Il n'est pas difficile de généraliser à 3D une fois qu'on a validé que le 2D fonctionne.
        De la meme facon, je n'ai pas validé la reponse obtenue, ce sera fait p[lus tard.

        """

        # Define a 3d array (float32)
        d, h, w = 256, 256, 256
        size = d * w * h
        host_array = np.zeros(shape=(d, h, w), dtype=np.float32)
        host_array[0, :, :] = 10

        # Get OpenCL platform and device
        platform = cl.get_platforms()[0]  # Select first platform
        device = platform.get_devices()[0]  # Select first device (GPU or CPU)
        context = cl.Context([device])  # Create OpenCL context
        queue = cl.CommandQueue(context)  # Create command queue

        # Create OpenCL buffers
        d_input = cl_array.to_device(queue, host_array)  # Copy data to GPU
        d_output = cl_array.empty_like(d_input)  # Create an empty GPU array

        # Kernel program: Laplace in 3D
        # Build the program
        program = cl.Program(context, self.kernel_code).build()

        # Set up the execution parameters
        global_size = (w, h, d)  # Matches the 2D array size

        start_time = time.time()
        for i in range(5000):
            program.laplace3D(
                queue,
                global_size,
                None,
                d_input.data,
                d_output.data,
                np.int32(w),
                np.int32(h),
                np.int32(d),
            )
            # The calculation is sent to d_output, which I then use as the input for another iteration
            # This way, d_input becomes the output and I do not have to create an array each time.  This is very efficient.
            program.laplace3D(
                queue,
                global_size,
                None,
                d_output.data,
                d_input.data,
                np.int32(w),
                np.int32(h),
                np.int32(d),
            )

            if i % 100 == 0:  # On verifie la convergence, mais pas tout le temps.
                d_diff = d_output - d_input

                mean_val = (
                    cl_array.sum(d_diff).get() / size
                )  # Mean (transfers single float)
                d_diff_sq = (d_diff - mean_val) ** 2  # Element-wise (x - mean)^2
                variance_val = (
                    cl_array.sum(d_diff_sq).get() / size
                )  # Variance (transfers single float)
                max_val = cl_array.max(d_diff).get()
                min_val = cl_array.min(d_diff).get()
                std_val = np.sqrt(variance_val)  # Standard deviation (final sqrt)
                if std_val < TOLERANCE:
                    break

        print(
            f"\nOpenCL, 3D single grid {host_array.shape}: {time.time() - start_time:.3f}"
        )

        # Retrieve results
        output_array = d_output.get()

        for i in range(0, output_array.shape[0], 10):
            plt.title(f"{self._testMethodName} Depth: {i}")
            plt.imshow(output_array[i, :, :])
            plt.pause(0.1)

    def setUp(self):
        # Get OpenCL platform and device
        self.platform = cl.get_platforms()[0]  # Select first platform
        self.device = self.platform.get_devices()[0]  # Select first device (GPU or CPU)
        self.context = cl.Context([self.device])  # Create OpenCL context
        self.queue = cl.CommandQueue(self.context)  # Create command queue

        # Build the program
        self.program = cl.Program(self.context, self.kernel_code).build()

    def tearDown(self):
        super().tearDown()

    def test03_scale_numpyarray(self):
        """
        Une des facons d'accelerer le calcul est de faire des raffinements successifs
        de la resolution. Je vais tenter de coder cela sur le GPU

        """

        # Define a 3d array (float32)
        h, w = 32, 32
        size = w * h
        host_array = np.random.randint(0, 255, size=(h, w)).astype(np.float32)
        host_dest = np.zeros(shape=(2 * h, 2 * w), dtype=np.float32)

        # Create OpenCL buffers
        d_input = cl_array.to_device(self.queue, host_array)  # Copy data to GPU
        d_output = cl_array.to_device(
            self.queue, host_dest
        )  # Create an empty GPU array

        # Set up the execution parameters
        global_size = (w, h)  # Matches the 2D array size

        self.program.zoom2D_nearest_neighbour(
            self.queue,
            global_size,
            None,
            d_input.data,
            d_output.data,
            np.int32(w),
            np.int32(h),
        )

        # Retrieve results
        input_array = d_input.get()
        output_array = d_output.get()

        # Validate answer
        expected_output_array = zoom(input_array, 2, order=0)

        self.assertEqual(input_array.shape[0] * 2, output_array.shape[0])
        self.assertEqual(input_array.shape[1] * 2, output_array.shape[1])

        self.assertEqual(expected_output_array.shape[0], output_array.shape[0])
        self.assertEqual(expected_output_array.shape[1], output_array.shape[1])

        self.assertTrue((expected_output_array == output_array).all())

        plt.title(f"{self._testMethodName}")
        plt.imshow(output_array)
        plt.pause(0.1)

    def test04_scale_numpyarray_twice(self):
        """
        Je valide que je peux facilement zoomer a repetition
        """

        # Define a 3d array (float32)
        h, w = 32, 32
        size = w * h
        host_array = np.random.randint(0, 255, size=(h, w)).astype(np.float32)

        # Create OpenCL buffers
        d_input = cl_array.to_device(self.queue, host_array)  # Copy data to GPU

        for s in range(3):
            host_dest = np.zeros(shape=(2 * h, 2 * w), dtype=np.float32)
            d_output = cl_array.to_device(
                self.queue, host_dest
            )  # Create an empty GPU array

            self.program.zoom2D_nearest_neighbour(
                self.queue,
                (w, h),
                None,
                d_input.data,
                d_output.data,
                np.int32(w),
                np.int32(h),
            )
            h *= 2
            w *= 2

            d_input = cl_array.empty_like(d_output)
            self.program.copy(
                self.queue, (w, h), None, d_output.data, d_input.data, np.int32(w)
            )

        # # Retrieve results
        input_array = d_input.get()
        output_array = d_output.get()

        plt.title(f"{self._testMethodName}")
        plt.imshow(input_array)
        plt.pause(0.1)

    def test05_laplace2d_grid_refinement(self):
        """
        Je valide que je peux facilement zoomer a repetition en faisant Laplace.
        Les tests ne sont pas tres concluant car je dois reallouer de la memoire a repetition
        ce qui coute tres cher en performance.

        Il faudrait eviter de réallouer de la memoire a chaque iteration en
        ayant des numpy array plus grands que necessaires mais cela
        demanderait de calculer les positions dans les numpy array a la
        main partout.  Pour l'instant, c'est trop de travail sans être certain
        que le calcul sera énormement plus rapide.
        """

        # Define a 3d array (float32)
        h, w = 32, 32
        size = w * h
        host_array = np.zeros(shape=(h, w), dtype=np.float32)
        host_array[0, :] = 10

        # Create OpenCL buffers
        d_input = cl_array.to_device(self.queue, host_array)  # Copy data to GPU

        start_time = time.time()
        for s in range(2):
            host_dest = np.zeros(shape=(h, w), dtype=np.float32)
            d_output = cl_array.to_device(self.queue, host_dest)

            for i in range(5000):
                # The calculation is sent to d_output, which I then use as the input for another iteration
                # This way, d_input becomes the output and I do not have to create an array each time.  This is very efficient.
                self.program.laplace(
                    self.queue, (w, h), None, d_input.data, d_output.data, np.int32(w)
                )
                self.program.laplace(
                    self.queue, (w, h), None, d_output.data, d_input.data, np.int32(w)
                )

                if i % 100 == 0:
                    d_diff = d_output - d_input

                    mean_val = (
                        cl_array.sum(d_diff).get() / size
                    )  # Mean (transfers single float)
                    d_diff_sq = (d_diff - mean_val) ** 2  # Element-wise (x - mean)^2
                    variance_val = (
                        cl_array.sum(d_diff_sq).get() / size
                    )  # Variance (transfers single float)
                    max_val = cl_array.max(d_diff).get()
                    min_val = cl_array.min(d_diff).get()
                    std_val = np.sqrt(variance_val)  # Standard deviation (final sqrt)
                    if std_val < TOLERANCE:
                        break

            zoom_dest = np.zeros(shape=(2 * h, 2 * w), dtype=np.float32)
            d_input = cl_array.to_device(self.queue, zoom_dest)

            self.program.zoom2D_nearest_neighbour(
                self.queue,
                (w, h),
                None,
                d_output.data,
                d_input.data,
                np.int32(w),
                np.int32(h),
            )
            h *= 2
            w *= 2

        input_array = d_input.get()
        output_array = d_output.get()

        print(
            f"\nOpenCL, 2D grid refinement {input_array.shape}: {time.time() - start_time:.3f}"
        )

        # # Retrieve results

        plt.title(f"{self._testMethodName}")
        plt.imshow(input_array)
        plt.pause(0.1)

    @property
    def kernel_code(self):
        kernel_code = """

        __kernel void laplace(__global float* input, __global float* output, int width) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int index = y * width + x;

            if (x == 0 || y == 0 || x == width-1 || y == width-1) {
                output[index] = input[index]; // Boundary is fixed
            } else {
                output[index] = (input[index-1] + input[index+1] + input[index-width] + input[index+width])/4;
            }
        }

        __kernel void laplace3D(__global float* input, __global float* output, int width, int height, int depth) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            int z = get_global_id(2);

            int index = z * (width * height) + y * width + x;

            if (x == 0 || y == 0 || z == 0 || x == width-1 || y == height-1 || z == depth-1) {
                output[index] = input[index];
            } else {
                output[index] = (input[index-1] + input[index+1] + input[index-width] + input[index+width] + input[index-width*height] + input[index+width*height])/6;
            }
        }


        __kernel void zoom2D_nearest_neighbour(__global float* input, __global float* output, int width, int height) {
            int x = get_global_id(0);
            int y = get_global_id(1);

            int index_src = y * width + x;

            int index_dest1 = (2*y) * (2*width) + 2*x;
            int index_dest2 = (2*y) * (2*width) + 2*x + 1;
            int index_dest3 = (2*y) * (2*width) + 2*x + 2*width;
            int index_dest4 = (2*y) * (2*width) + 2*x + 2*width + 1;

            output[index_dest1] = input[index_src];
            output[index_dest2] = input[index_src];
            output[index_dest3] = input[index_src];
            output[index_dest4] = input[index_src];
        }

        __kernel void copy(__global float* input, __global float* output, int width) {
            int x = get_global_id(0);
            int y = get_global_id(1);

            int index = y * width + x;

            output[index] = input[index];
        }

        """
        return kernel_code


if __name__ == "__main__":
    # unittest.main(defaultTest=["OpenCLArrayTestCase.test05_laplace2d_grid_refinement","OpenCLArrayTestCase.test01_2Dopencl","ArrayManipulationTestCase.test12_laplace_with_finer_and_finer_grid","ArrayManipulationTestCase.test10_laplace_initial_condition_fct"])
    unittest.main()

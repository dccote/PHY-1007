# PHY-1007
Code associé au cours PHY-1007 : Électromagnétisme à l'Université Laval. Le répertoire est organisé en `Semaine-XX` pour s'arrimer au [site Web du cours PHY-1007](https://sitescours.monportail.ulaval.ca/ena/site/accueil?idSite=174289&idPage=4617116).

Pour commencer, les exemples demandent simplement de connaître les noms des commandes pour créer des graphiques (champs scalaires, vectoriels, etc.).

## Semaine-01 : 

Du code réalisé par Louis Archambault pour vous donner des exemples de ce qui est possible en termes de visualisation.

## Semaine-02 :

Du code réalisé par [Daniel C. Côté](https://github.com/dccote) pour vous guider pas à pas et vous montrer une progression dans un code qui commence très simplement et qui s'améliore itérativement. J'ai bien l'intention de continuer à améliorer ce code chaque semaine. Lisez les fichiers dans l'ordre des numéros pour bien suivre la progression.

* **Visualisation.ipynb** : un notebook facile à utiliser pour les débutants. Si vous avez Jupyter sur votre ordinateur (`pip install notebook`), lancez simplement `jupyter notebook`. Sinon, cliquez sur ce lien pour [l'exécuter sur Google Colab](https://colab.research.google.com/github/dccote/PHY-1007/blob/master/Semaine-02/Visualisation.ipynb).

* **01 à 05 :** Vous avez ici le code Python le plus simple pour afficher une flèche sur un graphique et, à la fin, du code qui affiche un champ dans le plan. Pour du code procédural, c'est ce qu'on peut faire de mieux pour générer un graphique en peu de lignes. Vous verrez que le code nécessite malgré tout une gestion triviale de différents aspects et, en dépit de sa simplicité, reste assez difficile à réutiliser : il faut essentiellement recopier le code.

* **06 à 11 :** Pour mieux réutiliser le code, la solution consiste à créer une classe qui encapsule les détails et les gère. Vous verrez une utilisation de la programmation orientée objet pour créer des classes et simplifier l'utilisation à long terme dans d'autres projets. Je vais continuer à développer ce code tout au long de la fin de semaine.

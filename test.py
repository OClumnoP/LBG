import numpy as np

# Création d'un tableau 3D
tableau = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

# L'élément à ajouter
new_row = np.array([[9, 10]])

# Vérifiez la forme du tableau avant de modifier
print("Forme avant:", tableau.shape)

# Utilisation de np.insert pour ajouter la nouvelle ligne à tableau[0]
tableau_modifie = np.insert(tableau, tableau.shape[1], new_row, axis=1)

# Afficher la forme et le contenu du tableau après modification
print("Forme après:", tableau_modifie.shape)
print("Tableau modifié:\n", tableau_modifie)

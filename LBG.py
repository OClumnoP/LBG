from typing import Any

from PIL import Image
import numpy
import cupy as np #On a remplacé les tableaux numpy par des tableaux cupy, pour compute LBG avec notre GPU, et pas le CPU
import cv2
import time
import os
import pickle


# image_to_gray_matrix convertit une image en une matrice de niveau de gris
def image_to_gray_matrix(image_path):
    # Charge l'image à partir de son chemin
    image = Image.open(image_path)
    # Vérifie si l'image n'est pas déja en niveau de gris ; sinon, elle la converti
    if image.mode != 'L':
        image = image.convert('L')
    # Converti l'image en niveau de gris vers une matrice de niveau de gris et retourne cette matrice
    image_matrix = np.array(image)
    return image_matrix

# Prend en entrée une matrice, la décompose en bloc de taille NxN, mappe chaque bloc en un vecteur, et retourne la liste des vecteurs associés
def matrice_en_blocs_colonnes(matrice, N):
    # Vérification des dimensions de la matrice
    if matrice.shape[0] % N != 0 or matrice.shape[1] % N != 0:
        raise ValueError("La taille de la matrice doit être un multiple de N")

    # Initialiser la liste des vecteurs
    vecteurs = []

    # Parcourir la matrice par blocs de taille NxN
    for i in range(0, matrice.shape[0], N):
        for j in range(0, matrice.shape[1], N):
            # Extraire le bloc
            bloc = matrice[i:i + N, j:j + N]
            # Convertir le bloc en vecteur en empilant les colonnes
            vecteur = bloc.T.flatten()
            # Ajouter le vecteur à la liste
            vecteurs.append(vecteur)

    return vecteurs

def center_of_gravity(vectors):
    vecteurs_non_nuls = vectors[np.any(vectors != 0, axis=1)]
    moyenne_vecteurs = np.mean(vecteurs_non_nuls, axis=0)
    return moyenne_vecteurs


# Prend en entrée un vecteur, et retourne une liste de deux vecteurs déplacés de eps. (eps est un petit vecteur de la meme dimension que vector)
def split_vector(vector, eps):
    return np.array([vector + eps, vector - eps])


# Renvoie la distance euclienne entre deux vecteurs v1 et v2
def distance_euclienne(v1, v2):
    return np.linalg.norm(v1 - v2)

# À partir d'un vecteur et d'une liste de classes, cette fonction renvoie l'indice (dans la liste classes) de la classe à laquelle appartient ce vecteur, selon le critère de la minimisation de la norme euclidiennne
def class_vector(vector, classes):

    indice = np.argmin(np.sqrt(np.sum((classes - vector) ** 2, axis=1)))
    return indice


# Prend en entree un ensemble de clusters, chaque clusters contenant des vecteurs, la liste de leurs centres de gravité associés centers. Calcule la distortion moyenne sur l'ensemble des clusters par rapport à leur centre de gravité respectif.
def distortion_moyenne(clusters, centers):
    total_distortion = 0
    total_vectors = 0
    nb_clusters = clusters.shape[0]  # nombre de clusters
    for k in range(nb_clusters):  # Parcours chaque cluster
        cluster = clusters[k]
        center = centers[k]  # le centre de gravité associé au cluster k

        for vector in cluster:  # parcours l'ensemble des vecteurs contenus dans le cluster k et calcule la distortion moyenne sur celui ci
            if not np.all(vector == 0).item():  # Ignore les vecteurs nuls, car on travaille avec des tableaux numpy de taille non modulables
                distance = np.linalg.norm(vector - center) ** 2
                total_distortion += distance
                total_vectors += 1  # compte le nombre de vecteurs intervenants dans le calcul de la distortion

    if total_vectors == 0:
        return 0  # Evite la division par zero. Si tous les vecteurs sont nuls, alors les centre de gravité sont tous nuls et la distortion est nulle
    average_distortion = total_distortion / total_vectors
    return average_distortion


# Effectue l'opération inverse du formattage. La fonction remappe les vecteurs en blocs, merge les blocs en une matrice de niveau de gris, puis converti en image cette matrice.
def vecteurs_en_image(vecteurs, largeur, hauteur, N):
    # Vérification des dimensions
    if largeur % N != 0 or hauteur % N != 0:
        raise ValueError("La largeur et la hauteur doivent être des multiples de N")

    # Initialiser une matrice de zéros pour l'image
    image = np.zeros((hauteur, largeur), dtype=np.uint8)

    # Calculer le nombre de blocs par ligne et par colonne
    nb_blocs_ligne = largeur // N
    nb_blocs_colonne = hauteur // N

    # Parcourir les vecteurs
    idx_vecteur = 0
    for i in range(nb_blocs_colonne):
        for j in range(nb_blocs_ligne):
            # Extraire le vecteur
            vecteur = vecteurs[idx_vecteur]
            # Reshape pour revenir à la forme de bloc NxN
            bloc = vecteur.reshape(N, N)
            # Remplacer les pixels dans l'image
            image[i * N:(i + 1) * N, j * N:(j + 1) * N] = bloc
            idx_vecteur += 1

    # Enregistrer l'image
    nom_fichier = "C:/Users/Axel/LBG/reconstructedImageTest.png"
    cv2.imwrite(nom_fichier, image)

    return image

# Fonction principale. Prend en entrée un chemin d'image, la taille de bloc voulue (typiquement 8x8 ou 16x16), le seuil de convergence voulu pour la décroissance de la distortion et le nombre de prototypes voulu pour le codebook.
def LBG(database, N, delta, n):  # N: taille de bloc voulue ; delta: distortion decrease max threshold ; n: nombre voulu de vecteurs prototypes (multiple de 2).
    start_alg=time.time()
    print("démarrage de l'algorithme de LBG")
    vectors = []
    for image in os.listdir(database):  # Parcourt toutes les images du dossier
        image_path = os.path.join(database, image)
        matrix = image_to_gray_matrix(image_path)  # Charge l'image
        vectors += matrice_en_blocs_colonnes(matrix, N)  # La converti en vecteurs de niveau de gris
    vectors_array = np.array(vectors)  # Converti la liste des vecteurs en un tableau pour faciliter les opérations
    vector_size = vectors_array.shape[1]
    nb_vectors = vectors_array.shape[0]
    print("Nombre de vecteurs dans notre pool", nb_vectors)
    eps = np.random.uniform(0.1, 0.1, N * N)  # Défini le petit vecteur de deplacement epsilon
    c0 = center_of_gravity(vectors_array)  # Initialise le centre de gravité de ma liste de vecteurs
    prototypes = np.array([c0])  # Initialise le dictionnaire
    print("Dictionnaire initialisé avec le centre de gravité du pool")
    count_step = 0
    while prototypes.shape[0] < n:  # Répéter jusqu'à obtenir assez de vecteurs prototypes
        count_step += 1
        print("Nombre de vecteurs prototypes au démarrage de l'étape "+str(count_step)+" :", prototypes.shape[0])
        classes = []  # On initialise la liste des classes
        for vector in prototypes:  # On va diviser chaque vecteur prototype en deux vecteurs
            classes.append(split_vector(vector, eps)[0])
            classes.append(split_vector(vector, eps)[1])
        prototypes = np.array(classes)
        print("Nombre de vecteurs prototypes après propagation :", prototypes.shape[0])
        nb_prototypes = prototypes.shape[0]
        mean_distortion_before = 1e9  # On initialise les distortions de sorte à pouvoir entrer dans la boucle while
        mean_distortion_after = mean_distortion_before - 1
        mean_distortion_dec = np.abs(mean_distortion_after - mean_distortion_before)
        count_cvg = 0
        start_cvg = time.time()
        while mean_distortion_dec >= delta:  # Jusqu'à ce que la décroissance de la distortion moyenne descende en dessous du seuil fixé
            if(count_cvg>=20): #On ne veut pas excéder 20 étapes de convergence, par soucis de durée computationelle
                print("On a atteint 20 étapes de convergence, on s'arrête ici pour l'étape", count_step)
                break
            count_cvg += 1
            print("Step de convergence", str(count_step)+"."+str(count_cvg))
            clusters = np.zeros((nb_prototypes, 0, vector_size))  # On initialise un tableau à 3 dimensions. Il y'a autant de couches qu'il y'a de classes possibles ; cela permet de classer les vecteurs de l'image dans la classe qui convient. Au sein d'une couche (un cluster), chaque ligne correspond à un vecteur.
            count_vect = 0
            start_classification = time.time()
            for vector in vectors_array:  # On va classer chaque vecteurs dans un cluster, selon la classe à laquelle il appartient
                count_vect += 1
                if count_vect % 1024 == 0:
                    print("Traitement du vecteur", count_vect)
                classe = class_vector(vector, prototypes)

                #On agrandit le tableau et on y insère le nouveau vecteur
                new_shape = (clusters.shape[0], clusters.shape[1] + 1, clusters.shape[2])
                new_clusters = np.zeros(new_shape, dtype=clusters.dtype)
                new_clusters[:, :-1, :] = clusters
                clusters = new_clusters
                clusters[classe, -1, :] = vector

                '''
                for i in range(nb_vectors):
                    if np.all(clusters[classe, i, :] == 0).item(): #On cherche la première position libre dans le cluster numéro i
                        clusters[classe, i, :] = vector #On a trouvé cette position, on peut y mettre notre vecteur #Pour une raison inconnue, cette ligne ne fonctionne pas. On ne peut donc pas utiliser cette méthode d'insertion
                        break  # Sortir de la boucle dès qu'une position libre est trouvée
                '''
            print("On a fini de classer les clusters pour l'étape ", count_step, ".", count_cvg)
            end_classification = time.time()
            print("Durée de la classification à l'étape", str(count_step)+"."+str(count_cvg), ":", end_classification - start_classification, "secondes")

            for k in range(clusters.shape[0]):  # On calcule le centre de gravité de chaque cluster, qui remplacent les anciens prototypes
                prototypes[k] = center_of_gravity(clusters[k])
            print("On a remplacé les anciens prototypes par les centres de leur cluster pour l'étape", str(count_step)+"."+str(count_cvg))

            mean_distortion_before = mean_distortion_after  # On retient l'ancienne distortion moyenne
            print("Ancienne distortion", mean_distortion_before)
            mean_distortion_after = distortion_moyenne(clusters,
                                                       prototypes)  # On calcule la nouvelle, pour accéder à la décroissance
            print("Nouvelle distortion", mean_distortion_after)
            mean_distortion_dec = np.abs(mean_distortion_after - mean_distortion_before)
            print("Valeur de la décroissance", mean_distortion_dec)
        end_cvg = time.time()
        print("Durée de l'étape"+str(count_step)+" :", end_cvg-start_cvg)
    end_alg=time.time()
    print("Durée totale de l'algorithme :", end_alg-start_alg)
    return prototypes

def quantify_image(image_path, prototypes):
    matrix = image_to_gray_matrix(image_path)  # Charge l'image
    if matrix.shape[0] != matrix.shape[1]:
        return "L'image n'est pas carrée !"
    # Vérifie si la taille est une puissance de 2
    taille = matrix.shape[0]
    if (taille & (taille - 1)) != 0:
        return "La taille de l'image n'est pas une puissance de 2 !"
    img_size = matrix.shape[0]
    N = int(np.sqrt(prototypes.shape[1]))
    vectors = matrice_en_blocs_colonnes(matrix, N)  # Convertie la matrice en vecteurs de niveau de gris
    vectors_array = np.array(vectors)  # Converti la liste des vecteurs en un tableau pour faciliter les opérations
    nb_vectors = vectors_array.shape[0]
    for k in range(nb_vectors):  # On parcours la liste des vecteurs de l'image de départ
        vectors_array[k] = prototypes[class_vector(vectors_array[k],
                                                   prototypes)]  # On remplace chaque vecteur par son prototype le plus proche, au sens de la distance euclidienne
    reconstructed_image = vecteurs_en_image(vectors_array, img_size, img_size,
                                            N)  # On reconstruit l'image obtenue à partir du dictionnaire de prototypes.
    return reconstructed_image

def main():
    db16_small = "C:/Users/Axel/LBG/db16_small" #On charge la base de données
    prototypes1024 = LBG(db16_small, 16, 0.1, 1024) #On assigne à une variable le dictionnaire généré par l'algorithme de LBG
    with open('C:/Users/Axel/LBG/dictionary1024.txt', 'wb') as f: #On enregistre le dictionaire dans un un fichier txt pour le conserver
        pickle.dump(prototypes1024, f)
    with open('C:/Users/Axel/LBG/dictionary1024.txt', 'rb') as f: #On peut assigner à une variable le dictionnaire enregistré dans le fichier txt
        unpickled = pickle.load(f)
    quantify_image("C:/Users/Axel/LBG/chatTest.png", unpickled) #On peut quantifier une image avec le dictionnaire qu'on a crée
if __name__ == '__main__':
    main()

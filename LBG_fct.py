import math
from functools import reduce
from collections import defaultdict
import random
import numpy as np

data_size = 0
dim = 0


def genCodebook(data, N, delta=0.5):  # delta = seuil du max de distortion moyenne
    global data_size
    global dim

    data = np.array(data)
    data_size = len(data)
    assert data_size > 0

    dim = data.shape[1]
    assert dim > 0

    codebook = []
    codebook_poids_abs = np.array([data_size])
    codebook_poids_rel = np.array([1.0])

    # Calcul du vecteur initial du codebook : on prend la moyenne de tous les vecteurs de notre image
    c0 = moy_vec_de_vecs(data, dim, data_size)
    codebook.append(c0)

    # Calcul de la distorsion moyenne
    dist_moy = distortion_moy_c0(c0, data)

    # On split les "codevectors" autant que nécessaire
    while len(codebook) < N:
        codebook, codebook_poids_abs, codebook_poids_rel, dist_moy = split_codebook(data, codebook, dist_moy, delta)

    return np.array(codebook), np.array(codebook_poids_abs), np.array(codebook_poids_rel)


def moy_vec_de_vecs(vecs, dim_vecs=None, taille_vecs=None):
    vecs = np.array(vecs)
    dim_vecs = dim_vecs or vecs.shape[1]
    taille_vecs = taille_vecs or vecs.shape[0]
    vec_moy = np.mean(vecs, axis=0)
    return vec_moy


def distortion_moy_c0(c0, data, size=None):
    size = size or data_size
    return np.mean([distance_eucli(c0, vec) for vec in data])


def distortion_moy_codevector_list(codevector_list, data):
    return np.mean(
        [distance_eucli(codevector, vec) for codevector, vec in zip(codevector_list, data) if codevector is not None])


def distance_eucli(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


def gen_vec_epsilon(codevector):
    return np.random.uniform(-0.1, 0.1, size=len(codevector))


def split_codebook(data, codebook, moy_dist_initial, delta=0.5):
    # split codevectors
    new_codevectors = []
    for codevector in codebook:
        epsilon = gen_vec_epsilon(codevector)
        # On crée les deux nouveaux codevectors en adjutant et en soustrayant epsilon à chaque composante du codevector
        c1 = np.array(codevector) + epsilon
        c2 = np.array(codevector) - epsilon
        new_codevectors.extend((c1, c2))

    codebook = np.array(new_codevectors)
    len_codebook = len(codebook)
    poids_abs = np.zeros(len_codebook)
    poids_rel = np.zeros(len_codebook)

    # on cherche la convergence en minimisant la distorsion moyenne
    moy_dist = 0
    err = delta + 1
    num_iter = 0
    dist_moy = moy_dist_initial  # Initialisation de dist_moy

    while err > delta:
        # On cherche les codevectors les plus proches pour chaque vecteur de données (on trouve la proximité de chaque codevector)
        plus_proche_codevector_list = [None] * data_size
        vecs_proche_codevector = defaultdict(list)  # On mappe les vecteurs de données
        vec_idxs_proche_codevector = defaultdict(list)  # de même
        for i, vec in enumerate(data):  # pour chaque vecteur en entrée
            dist_min = None
            plus_proche_codevector_index = None
            for i_c, c in enumerate(codebook):  # pour chaque codevector
                d = distance_eucli(vec, c)
                if dist_min is None or d < dist_min:  # Trouver le nouveau codevector le plus proche
                    dist_min = d
                    plus_proche_codevector_list[i] = c
                    plus_proche_codevector_index = i_c
            vecs_proche_codevector[plus_proche_codevector_index].append(vec)
            vec_idxs_proche_codevector[plus_proche_codevector_index].append(i)

        # on met à jour le codebook : on recalcule chaque codevector pour qu'il se trouve au centre des points de proximité
        for i_c in range(len_codebook):  # pour chaque index de codevector
            vecs = vecs_proche_codevector.get(i_c) or []  # obtenir les vecteurs proches de ce codevector
            num_vecs_proche_codevector = len(vecs)
            if num_vecs_proche_codevector > 0:
                new_codevector = moy_vec_de_vecs(vecs, dim)  # On calcule le nouveau centre de ce codevector
                codebook[i_c] = new_codevector  # nouveau codevector dans codebook
                for i in vec_idxs_proche_codevector[i_c]:  # update in input vector index -> codevector mapping list
                    plus_proche_codevector_list[i] = new_codevector

                # update the weights
                poids_abs[i_c] = num_vecs_proche_codevector
                poids_rel[i_c] = num_vecs_proche_codevector / data_size

        # On calcule de nouveau la distorsion moyenne
        dist_moy_prec = dist_moy if dist_moy > 0 else moy_dist_initial
        dist_moy = distortion_moy_codevector_list(plus_proche_codevector_list, data)

        # On calcule de nouveau l'erreur
        err = (dist_moy_prec - dist_moy) / dist_moy_prec
        print(plus_proche_codevector_list)
        print('> iteration', num_iter, 'dist_moy', dist_moy, 'dist_moy_prec', dist_moy_prec, 'err', err)

        num_iter += 1

    return codebook, poids_abs, poids_rel, dist_moy


# Sample call to the function with test data
testdata = [[1, 2], [3, 4], [5, 6]]  # your test data
for codebook_size in (1, 2, 4, 8):
    print('Génération du codebook', codebook_size)
    codebook, codebook_p_abs, codebook_p_rel = genCodebook(testdata, codebook_size, delta=0.00001)
    print('output:')
    for i, c in enumerate(codebook):
        print(f'Codebook {i}: {c}')

"""
kdtree module
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from TD.nearest_neighbor import NearestNeighborSearch
from TD.nearest_neighbor import euclidean_distance as eucl_dist

def median(X: np.ndarray, start: int, stop: int, c: int) -> float:
    """
    Returns median of array X between indices start and stop for coordinate c
    """
    assert stop - start >= 0, "Requested stop > start"
    assert X.ndim == 2, "2D required"
    med = np.inf
    # Ex3
    arr = np.copy(X)
    arr = sorted(arr[start:stop], key=(lambda y : y[c]))
    med = arr[(stop-start)//2]
    return med[c]


def swap(X: np.ndarray, idx1, idx2) -> None:
    """Swaps two rows of a 2D numpy array"""
    X[idx1, :], X[idx2, :] = X[idx2, :], X[idx1, :]


def partition(X: np.ndarray, start: int, stop: int, c: int) -> int:
    """
    Partitions the array X between start and stop wrt to its median along a coordinate c
    """
    # Ex4
    # You may or may not use the median function above, up to you
    # med = median(X, start, stop, c)
    idx = -1
    valeur_mid = median(X, start, stop, c)  #Ici on récupère la valeur médiane qui servira pour la comparaison avec le pivot
    #print(f"La médiane de coordonnée {c} pour ce jeu de données vaut {valeur_mid}")        #Used for debugging

    #Méthode 1 : trier la liste. Petit problème, cela crée une autre liste
    #X = sorted(X[start:stop], key=(lambda y : y[c]))
    #for i in range(start,stop):
    #    print(f'{i} -> {X[i][c]}')
    #
    #for i in range(start, stop):
    #    if X[i][c] > valeur_mid:
    #        return i-1
    #return stop

    #Méthode 2 : on utilise simplement un pivot
    pivot_1 = start
    pivot_2 = start
    
    for i in range(start,stop):
        if X[i,c] < valeur_mid:   #(repère perso) Test d'écriture
            #On échange la valeur de i avec le pivot
            temp = np.copy(X[i])
            X[i] = X[pivot_1]
            X[pivot_1] = temp

            if pivot_1 != pivot_2:
                temp = np.copy(X[i])
                X[i] = X[pivot_2]
                X[pivot_2] = temp
            pivot_1+=1
            pivot_2+=1
        elif X[i,c] == valeur_mid:
            temp = np.copy(X[i])
            X[i] = X[pivot_2]
            X[pivot_2] = temp
            pivot_2+=1

    #Affichage du tableau séparé à la fin, utile pour le débuggage
    #for i in range(start,pivot_1):
    #    print(f'{X[i,c]}',end=',')
    #print('|||',end='')
    #for i in range(pivot_1,pivot_2):
    #    print(f'{X[i,c]}',end=',')
    #print('|||', end='')        
    #for i in range(pivot_2,stop):
    #    print(f'{X[i,c]}',end=',')
    #print()
    return pivot_2-1

@dataclass
class Node:
    idx: int
    med: float = np.inf
    c: int = 0
    left: Self = (
        None  # Self denotes that left (and right) is of same type as self == Node
    )
    right: Self = None


class KDTree(NearestNeighborSearch):
    def __init__(self, X):
        """
        Contrary to LinearScan, KDTree's constructor must build the tree
        To that end, we will loop through the coordinates of X,
        hence the need for the `dim` attribute below.
        """
        super().__init__(X)
        self.dim = X.shape[1]
        self.build()

    def _build(self, start: int, stop: int, c: int) -> Node | None:
        """
        Builds a node with a correct index by partitioning X along c between start and stop,
        including left and right children nodes
        """
        assert stop >= start, "Indices issue"
        if stop == start:
            return
        if stop == start + 1:
            return Node(start)
        next_c = (c + 1) % self.dim
        # Ex5: iteratively partition, retrieve median and recurse (while creating correct Nodes)
        #print(f"Etude du tableau de {start} à {stop}")     #Used for debugging
        index_median = partition(self.X ,start,stop,c) #On partitionne et on récupère l'index de la médiane
        retour = Node(index_median, self.X[index_median,c],c,self._build(start, index_median, next_c), self._build((index_median +1), stop, next_c))
        return retour

    def reset(self):
        """
        Resets current estimation of distance to and index of nearest neighbor
        """
        self._current_dist = np.inf
        self._current_idx = -1

    def build(self):
        """
        Builds the kdtree
        """
        self.reset()
        self.root = self._build(0, len(self.X), 0)

    def _defeatist(self, node: Node | None, x: np.ndarray):
        """
        Defeatist search of nearest neighbor of x in node
        """
        if node is None:
            return None
        if node.med > x[node.c]:
            retour = self._defeatist(node.left, x)
            if retour is None:  #Si le retour est None, cela veut dire qu'on est une feuille, auquel cas on renvoie notre l'indice de notre feuille
                return node.idx
        else:
            retour = self._defeatist(node.right, x)
            if retour is None:  #Si le retour est None, cela veut dire qu'on est une feuille, auquel cas on renvoie notre l'indice de notre feuille
                return node.idx

    def _backtracking(self, node: Node | None, x: np.ndarray):
        """
        Backtracking search of nearest neighbor of x in node
        """
        index, dist = self._backtracking_aux(node, x)
        return index

    def _backtracking_aux(self, node: Node | None, x: np.ndarray):
        """
        Backtracking search of nearest neighbor of x in node
        """
        if node is None:
            return (None,0)
        if node.med > x[node.c]:    #On cherche à gauche ou en bas pour la 2D
            #On cherche du côté où se trouve x en premier
            index,dist = self._backtracking_aux(node.left, x)
            if index is None:  #Si le retour est None, cela veut dire qu'on est une feuille, auquel cas on renvoie notre l'indice de notre feuille
                return node.idx, eucl_dist(self.X[node.idx], x)
            
            #Si il peut se trouver de l'autre côté alors on va le chercher
            if x[node.c] + dist >= node.med:
                index_prim,dist_prim = self._backtracking_aux(node.right, x)
                if dist_prim < dist:
                    return index_prim,dist_prim

            return index,dist
        else:       #On cherche à droite ou en haut pour la 2D
            #On cherche du côté où se trouve x en premier
            index,dist = self._backtracking_aux(node.right, x)
            if index is None:  #Si le retour est None, cela veut dire qu'on est une feuille, auquel cas on renvoie notre l'indice de notre feuille
                return node.idx, eucl_dist(self.X[node.idx], x)
            
            #Si il peut se trouver de l'autre côté alors on va le chercher
            if x[node.c] - dist <= node.med:
                index_prim,dist_prim = self._backtracking_aux(node.left, x)
                if dist_prim < dist:
                    return index_prim,dist_prim
            return index,dist
        
    def query(self, x, mode: str = "backtracking"):
        """
        Queries given mode 'backtracking' or 'defeatist'
        """
        super().query(x)
        self.reset()
        if mode == "defeatist":
            self._defeatist(self.root, x)
        elif mode == "backtracking":
            self._backtracking(self.root, x)
        else:
            raise ValueError("Incorrect mode!")
        return self._current_dist, self._current_idx

    def set_xaggle_config(self):
        self.mode = None  # Choose search strategy for xaggle
        pass

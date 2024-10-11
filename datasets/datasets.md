Les datasets contiennent des résultats d'exécution d'un programme avec plusieurs configurations.

Ce programme est un benchmark pour le noyau dgetrf (factorisation LU) de la librairie LAPACK.
Même si ce benchmark utilise des meta-repetitions, on estime un bruit de mesure d'environ 10 %.

La colonne "performance" est l'objectif que l'on souhaite modéliser.
Les colonnes "matrix_size_(n|m)" représente les tailles de la matrice sur lequel le calcul est effectué (n*m)
Les autres colonnes représentent divers leviers d'optimisations du noyau.
Leurs noms on été obfusqué volontairement.


L'ensemble des paramètres sont contenus dans :

"p1": [8, 256]
"p2": [0, 256]
"p3": [1, 32]
"p4": [0, 32]
"p5": [1, 32]
"p6": [0, 32]
"p7": [1, 99]
"p8":[1, 99]

"matrix_size_n": [1000, 5000]
"matrix_size_m": [1000, 5000]


"performance" :[0, +Inf]
import numpy as np
from modelisation import JSSP
from dfs import JSSPSolver
from dataset_reader import DatasetReader

# Afficher toutes les valeurs des matrices
np.set_printoptions(threshold=np.inf)

# Spécifier le chemin du fichier dataset.txt
file_path = r"C:\Users\HP\Desktop\S2I\S2\META H\Metaheuristique-Particle-Swarm-Optimisation\datasets\tai20_15.txt"

# Lire le fichier de données
dataset_reader = DatasetReader(file_path)
dataset_reader.read_file()

# Vérifier qu'on a bien chargé au moins une instance
instances = dataset_reader.get_all_instances()
if not instances:
    print("Erreur : Aucune instance n'a été chargée.")
    exit(1)

# Récupérer la première instance
first_instance = instances[0]  # On prend la première instance chargée

machines_matrix = first_instance["machines_matrix"]
times_matrix = first_instance["times_matrix"]

# Vérifier que les matrices sont bien chargées
print("Dimensions de la matrice des machines :", machines_matrix.shape)
print("Dimensions de la matrice des temps :", times_matrix.shape)

# Afficher les matrices complètes
print("\nMatrice des machines :")
print(machines_matrix)

print("\nMatrice des temps :")
print(times_matrix)

# Créer une instance de JSSP
jssp = JSSP(machines_matrix, times_matrix)

# Résolution avec DFS
solver = JSSPSolver(jssp)
schedule, makespan = solver.solve()

# Affichage des résultats
print("\nMeilleur ordonnancement :", schedule)
print("Makespan optimal :", makespan)
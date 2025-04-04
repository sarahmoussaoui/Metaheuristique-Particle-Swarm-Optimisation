import numpy as np

class DatasetReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.instances = []  # Liste pour stocker les deux premières instances

    def read_file(self):
        """
        Lit le fichier de données et charge uniquement les deux premières matrices de temps et de machines.
        """
        with open(self.file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]  # Supprimer les lignes vides

        index = 0
        while len(self.instances) < 2 and index < len(lines):
            # Chercher la première ligne contenant uniquement des nombres
            while index < len(lines):
                try:
                    header = list(map(int, lines[index].split()))
                    if len(header) == 6:  # Vérifier qu'on a bien 6 valeurs attendues
                        break
                except ValueError:
                    index += 1  # Passer à la ligne suivante si ce n'est pas un header valide

            if index >= len(lines):
                break  # Stop si on atteint la fin du fichier

            # Extraire les informations générales
            nb_jobs, nb_machines, time_seed, machine_seed, upper_bound, lower_bound = header
            index += 1  # Passer à la ligne suivante

            # Recherche de la section "Times"
            while index < len(lines) and "Times" not in lines[index]:
                index += 1
            index += 1  # Aller à la première ligne de la matrice

            times_matrix = [list(map(int, lines[i].split())) for i in range(index, index + nb_jobs)]
            index += nb_jobs  # Passer à la suite

            # Recherche de la section "Machines"
            while index < len(lines) and "Machines" not in lines[index]:
                index += 1
            index += 1  # Aller à la première ligne de la matrice

            machines_matrix = [list(map(int, lines[i].split())) for i in range(index, index + nb_jobs)]
            index += nb_jobs  # Passer à la suite

            # Stocker l'instance
            self.instances.append({
                "nb_jobs": nb_jobs,
                "nb_machines": nb_machines,
                "time_seed": time_seed,
                "machine_seed": machine_seed,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "times_matrix": np.array(times_matrix),
                "machines_matrix": np.array(machines_matrix)
            })

    def get_instance(self, index):
        """Retourne l'une des deux premières instances lues."""
        if 0 <= index < len(self.instances):
            return self.instances[index]
        else:
            raise IndexError("Index hors limites. Seulement deux instances sont chargées.")

    def get_all_instances(self):
        """Retourne toutes les instances lues (2 maximum)."""
        return self.instances
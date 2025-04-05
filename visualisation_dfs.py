import matplotlib.pyplot as plt
import numpy as np

# Données (tailles et temps estimés)
tailles = ["2×2", "3×3", "4×4", "5×5" ]
temps_seconde = [0.001, 0.0185, 7.80, 20100]  

# Création du graphe
plt.figure(figsize=(9, 6))

# Courbe principale avec marqueurs
line, = plt.plot(tailles, temps_seconde, 
                marker='o', 
                markersize=8,
                linewidth=2,
                color='royalblue',
                label='Temps d\'exécution (s)')

# Ajout des étiquettes de données
for i, (xi, yi) in enumerate(zip(tailles, temps_seconde)):
    plt.text(xi, yi, 
             f'{yi:.3f}s' if yi < 0.1 else f'{yi:.1f}s',
             ha='center', 
             va='bottom' if i % 2 == 0 else 'top',
             fontsize=10)

# Configuration de l'échelle
plt.yscale('log')
plt.ylim(0.0008, 30000)  # Bornes ajustées pour inclure toutes les valeurs

# Personnalisation des axes et grille
ax = plt.gca()
ax.grid(True, which="both", linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Formatage des ticks
ax.yaxis.set_major_formatter(plt.ScalarFormatter())
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1,000", "10,000"])

# Titres et légendes
plt.title('Complexité Algorithmique - Temps d\'exécution en fonction de la taille du problème\n(Méthode DFS + Branch & Bound pour JSSP)',
          fontsize=14, pad=20)
plt.xlabel('Taille des instances (Jobs × Machines)', fontsize=12)
plt.ylabel('Temps (secondes - échelle logarithmique)', fontsize=12)
plt.legend(loc='upper left')

# Ajustements finaux
plt.tight_layout()
plt.show()

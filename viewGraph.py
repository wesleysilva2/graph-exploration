import networkx as nx
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg' se você tiver o PyQt5 instalado
import matplotlib.pyplot as plt
import pickle
import os

with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f:
    G = pickle.load(f)

print("Nós:", G.nodes())
print("Arestas:", G.edges())

nx.draw(G, with_labels=True)
plt.show()

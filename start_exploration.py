import os
import networkx as nx
import pickle
from GraphExplorationEnv import GraphExplorationEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Crie o grafo aqui ou carregue de algum lugar
with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f: # Carrega o grafo salvo
    G = pickle.load(f)

# Defina os valores conforme seu grafo
ACTIONS = max([len(list(G.neighbors(n))) for n in G.nodes])

# Cria o ambiente
env = GraphExplorationEnv(network=G, actions_amout=9) # Ajuste o número de ações conforme necessário

# (Opcional) Checa se o ambiente segue as especificações Gym
check_env(env, warn=True)

# Treinamento com DQN
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_graph_exploration/")

# Treinar o agente
model.learn(total_timesteps=10)

# Salvar modelo treinado
model.save("dqn_graph_exploration")

import gymnasium as gym
import pickle
import networkx as nx
import numpy as np
import random

# Essa é a classe do ambiente personalizado, onde o agente pode se mover em um grafo
class GraphExplorationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # network: Grafo do tipo networkx.Graph
    # actions_amount: Número de ações possíveis (em geral, o máximo de vizinhos que um nó pode ter)
    # stopClass: Classe de parada personalizada (opcional)
    # rewardClass: Classe de recompensa personalizada (opcional)
    # initial, target: nós de início e destino (se não passados, são escolhidos aleatoriamente)
    def __init__(self, network: nx.Graph, actions_amout: int, stopClass = None, rewardClass = None, initial = None, target = None):
        self.network = network 

        self.initial = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)] if initial is None else initial
        self.state = self.initial
        self.target = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)] if target is None else target

        # Garante que o nó inicial e o nó alvo sejam diferentes
        # Se o nó inicial e o nó alvo forem iguais, escolhe um novo nó alvo aleatório
        while(self.state == self.target and self.network.number_of_nodes != 1):
            self.target = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]


        self.stop = DefaultStopClass() if stopClass is None else stopClass 
        self.reward = DefaultReward() if rewardClass is None else rewardClass 

        # Define o espaço de ação e o espaço de observação
        # O espaço de ação é discreto, com o número de ações igual ao número de vizinhos do nó atual
        # O espaço de observação é um vetor de dois elementos: o nó atual e o nó alvo
        self.action_space = gym.spaces.Discrete(actions_amout)
        self.observation_space = gym.spaces.Box(low=0, high=np.array([self.network.number_of_nodes() - 1 , self.network.number_of_nodes() - 1]), shape=(2,), dtype=np.int64)

        self.count = 0
    
    # Reseta o ambiente, escolhendo um nó inicial e um nó alvo aleatórios
    # Sorteia novo estado inicial e destino
    def reset(self,seed=None):
        super().reset(seed=seed)
        self.initial = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]
        self.state = self.initial
        list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]


        while(self.state == self.target and self.network.number_of_nodes != 1):
            self.target = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]
        
        self.count = 0 # Reinicia o contador de passos

        obs = (self.state, self.target) # Retorna a observação inicial: uma tupla (estado atual, destino) 
        return obs, {}
    
    
    # Realiza um passo no ambiente, movendo-se para um nó vizinho
    def step(self,action):
        self.count += 1
        
        # possibleNextStates: são os vizinhos do estado atual (os possíveis próximos estados)
        possibleNextStates = list(self.network.neighbors(self.state)) 
        previousState = self.state

        if len(possibleNextStates) != 0: # Atualiza o estado se possível, se houver vizinhos
            self.state = possibleNextStates[action]

        # Usa a rewardClass e stopClass para calcular recompensa e término do episódio
        reward = self.reward.getReward(self.state, previousState, action, self.target, self.network)

        terminated = self.stop.isTerminated(self.state, previousState, action, self.target, self.network)

        obs = (self.state, self.target)

        # Retorna: observação, recompensa, se o episódio terminou, se o episódio foi truncado (False), e um dicionário com metadados (count de passos)
        return obs, reward, terminated, False, {"count" : self.count}

# Essa é a classe base para as classes de recompensa e parada
class RewardBaseClass():
    def getReward(self, state, previousState, action, target, graph):
        raise NotImplementedError

# Essa é a classe base para as classes de parada
class StopConditionBaseClass():
    def isTerminated(self, state, previousState, action, target, graph):
        raise NotImplementedError

# Essa é a classe padrão de recompensa, que calcula a recompensa com base no tempo total e na quantidade de viagens 
class DefaultReward(RewardBaseClass):
    def __init__(self) -> None:
        super().__init__()
        with open('./output/combined_sum_amount.pkl', 'rb') as f:
            self.waitTimeDict = pickle.load(f) # waitTimeDict[(u, v)] = (tempo de espera, quantidade)


    def getReward(self, state, previousState, action, target, graph):

        totalTime = self.waitTimeDict[(previousState, state)][0] # tempo total entre os dois nós
        amount = self.waitTimeDict[(previousState, state)][1] # quantidade de viagens entre os dois nós

        reward = -totalTime / amount # A recompensa padrão é negativa: -totalTime / amount, incentivando caminhos com menor tempo médio

        if state == target: # Se o agente chega no destino, dá um bônus gigante: +10.000.000
            reward += 10000000

        return reward

# Essa é a classe padrão de parada, que termina o episódio quando o agente chega ao nó alvo   
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target
    

if __name__ == "__main__":

    # episodes: número de episódios de treino/teste
    # env: ambiente (instância da classe GraphExplorationEnv)
    # alpha: taxa de aprendizado (quanto o agente aprende a cada iteração)
    # gamma: fator de desconto (quão importante é o futuro em relação ao presente)
    # epsilon: taxa de exploração (probabilidade de explorar vs. explorar o conhecimento atual)
    # is_training: se True, treina o agente; se False, carrega a tabela Q de um arquivo
    def run_q(episodes, env, alpha, gamma, epsilon, is_training=True):
        import time
        import matplotlib.pyplot as plt
        startTime = time.time() # Registra o tempo para garantir que a execução não ultrapasse 2 horas
        try:
            if(is_training): # Se for treinamento, começamos com Q-table vazia
                q = {}
            else: # Se não for treinamento, carregamos a Q-table de um arquivo
                f = open('q.pkl', 'rb')
                q = pickle.load(f)
                f.close()
                
            stepsPerEpisode = np.zeros(episodes) # Array para guardar quantos passos foram dados em cada episódio
            stepCount = 0 # contador de passos do episódio atual
        
            for i in range(episodes): # Inicia o laço de episódios de treino/teste
                
                # Reinicia o ambiente (env.reset()) e recebe o estado inicial (posição atual + alvo)
                state = env.reset()[0]
                start = state[0]
                target = state[1]
                terminated = False # Terminated marca se o episódio terminou (atingiu o alvo)

                while(not terminated): # Enquanto o episódio não terminar seguir

                    if state not in q: # Se esse estado (posição atual + alvo) ainda não está na Q-table, criamos entradas para ele
                        q[state] = {}
                        for act in range(len(list(env.network.neighbors(state[0])))):
                            q[state][act] = 0 # Cada entrada é associada a um possível vizinho do nó atual     

                    a = random.random() # Gera número aleatório    

                    if is_training and a< epsilon: # Se a < epsilon → agente explora: escolhe ação aleatória
                        n = list(env.network.neighbors(state[0]))
                        action = random.randint(0, max(0,len(n) -1)) # based on number of neighbors of position node
                    else: # Senão → agente explota: escolhe ação com maior valor Q (com desempate aleatório)           
                        values = [ q[state][act] for act in range(len(list(env.network.neighbors(state[0])))) ]
                        ix_max = [ act for act, value in enumerate(values) if value == max(values) ]
                        action = ( random.choice(ix_max) ) if len(ix_max) > 0 else 0
                    
                    # Faz o passo no ambiente, recebendo o novo estado, recompensa, se o episódio terminou (chegou ao alvo?), e metadados
                    new_state,reward,terminated,_,extra = env.step(action)
                    
                    # Mesma lógica que antes — se o novo estado ainda não está na Q-table, criamos entradas
                    if new_state not in q:
                        q[new_state] = {}
                        for act in range(len(list(env.network.neighbors(new_state[0])))):
                            q[new_state][act] = 0

                    # Atualização da Q-table (apenas se for treino)
                    # Essa é a equação padrão do Q-learning 
                    if is_training:
                        qValues = [ q[new_state][act] for act in range(len(list(env.network.neighbors(new_state[0])))) ]

                        sample = reward + gamma * (max(qValues) if len(qValues) > 0 else 0)

                        part1 = ( (1 - alpha) * q[state][action] ) if len(q[state]) > 0 else float('-inf')
                        part2 = alpha * sample
                        
                        q[state][action] = part1 + part2

                    # Atualiza o estado atual para o novo estado
                    state = new_state

                    stepCount += 1

                    # Se o episódio terminou (o agente chegou ao alvo), imprime o resultado e salva o número de passos
                    if terminated:
                        print(f"({i}) {start} -> {target} in {stepCount} steps with epsilon {epsilon}")
                        stepsPerEpisode[i] = stepCount
                        stepCount = 0
                
                # Reduz o epsilon ao longo dos episódios para explorar menos e explorar mais com o tempo
                epsilon = max(epsilon - 1/(episodes), 0.01) # Nunca deixa epsilon menor que 0.01 (exploração mínima)
                if time.time() - startTime > 60 * 60 * 2: # 2 hours
                    break

            env.close() # Fecha o ambiente

        finally:
            # Calcula a média móvel dos passos por episódio (últimos 100 episódios)
            # Plota esse gráfico e salva
            sumSteps = np.zeros(i)
            for t in range(i):
                sumSteps[t] = np.mean(stepsPerEpisode[max(0, t-100):(t+1)]) 

            plt.plot(sumSteps)
            plt.savefig(f'q{env.initial}-{env.target}-{alpha}-{gamma}-{epsilon}.png')

            # Se for treinamento, salva a Q-table em um arquivo
            if is_training:
                # Save Q Table
                f = open(f'q{env.initial}-{env.target}-{alpha}-{gamma}-{epsilon}.pkl',"wb")
                pickle.dump(q, f)
                f.close()


    with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f: # Carrega o grafo salvo
        G = pickle.load(f)

    
    env = GraphExplorationEnv(G, 9) # Cria o ambiente com o grafo carregado e 9 ações possíveis (número máximo de vizinhos de um nó)

    run_q(10, env, 0.9, 0.9, 0.1, is_training=True) # Executa run_q(...) com 1000 episódios e hiperparâmetros definidos (alpha=0.9, gamma=0.9, epsilon=0.1).

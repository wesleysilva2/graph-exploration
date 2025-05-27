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

        # Um delay dinamico em alguns nós, eventos que aumentam o tempo de viagem em certas arestas ao longo do tempo
        self.dynamicDelays = {} # dicionário que armazena o tempo de espera dinâmico em cada aresta (u, v) do grafo

        # Acumulador de tempo que o agente gastou
        self.estimated_time_so_far = 0

        # Tempo estimado do trajeto ótimo (útil para comparar e penalizar atrasos)
        self.max_expected_time = 0
    
    # Reseta o ambiente, escolhendo um nó inicial e um nó alvo aleatórios
    # Sorteia novo estado inicial e destino 
    def reset(self,seed=None):
        super().reset(seed=seed)
        self.initial = random.choice(list(self.network.nodes))
        self.target = random.choice(list(self.network.nodes))
        
        while self.target == self.initial and self.network.number_of_nodes() > 1: # Garante que o nó inicial e o nó alvo sejam diferentes
            self.target = random.choice(list(self.network.nodes))

        self.state = self.initial # Define o estado inicial como o nó inicial
        self.estimated_time_so_far = 0 # Reinicia o tempo estimado
        self.count = 0 # Reinicia o contador de passos

        self.generate_random_delay(self.initial,self.target) # Geração de atraso simulado em uma aresta aleatória

        # Definição de peso da aresta
        def edge_weight(u, v, d):
            key = (min(str(u), str(v)), max(str(u), str(v))) # Ordena os nós da aresta para evitar duplicação
            return self.reward.waitTimeDict.get(key, (1, 1))[0] # Pega o tempo de espera da aresta, se não existir, usa (1, 1) como padrão
        
        # Calcula o caminho mais curto entre o nó inicial e o nó alvo, considerando o tempo de espera
        # Dijkstra é usado para encontrar o caminho mais curto entre dois nós em um grafo ponderado
        try:
            path = nx.shortest_path(self.network, self.initial, self.target, weight=edge_weight) # Calcula o caminho mais curto entre o nó inicial e o nó alvo
            self.max_expected_time = sum( # Soma o tempo de espera de cada aresta no caminho
                self.reward.waitTimeDict.get((min(str(path[i]), str(path[i+1])), max(str(path[i]), str(path[i+1]))), (1, 1))[0] # Tempo de espera da aresta
                for i in range(len(path)-1)
            )
            print("Caminho ótimo entre o nó inicial e o alvo: ", path)
            print("Tempo estimado do trajeto ótimo: ", self.max_expected_time)
        
        except nx.NetworkXNoPath: # Se não houver caminho entre o nó inicial e o alvo, trata a exceção
            print("Nenhum caminho encontrado entre o node inicial e o alvo")
            self.max_expected_time = float('inf') # Define o tempo estimado como infinito, pois não há caminho
        
        while(self.state == self.target and self.network.number_of_nodes != 1): # Garante que o nó inicial e o nó alvo sejam diferentes
            self.target = list(self.network.nodes)[random.randint(0, self.network.number_of_nodes()-1)]

        obs = (self.state, self.target) # Retorna a observação inicial: uma tupla (estado atual, destino) 
        return obs, {}
    
    
    # Realiza um passo no ambiente, movendo-se para um nó vizinho
    def step(self,action):
        self.count += 1

        # possibleNextStates: são os vizinhos do estado atual (os possíveis próximos estados)
        possibleNextStates = list(self.network.neighbors(self.state)) 
        previousState = self.state

        if action >= len(possibleNextStates): # Se a ação escolhida for maior que o número de vizinhos, lança um erro
            raise ValueError(f"Ação {action} inválida. Apenas {len(possibleNextStates)} vizinhos disponíveis.")

        # Se o nó atual não tem vizinhos, termina o episódio
        if len(possibleNextStates) != 0: # Atualiza o estado se possível, se houver vizinhos
            self.state = possibleNextStates[action]

        # Antes de mudar de estado, calula-se o tempo real da aresta 
        # Aresta é uma tupla (u, v) onde u é o nó anterior e v é o nó atual
        # Aresta é ordenada para evitar duplicação, (u, v) e (v, u) serem considerados diferentes
        edge = (min(str(previousState), str(self.state)), max(str(previousState), str(self.state))) # Ordena os nós da aresta para evitar duplicação
        wait_time, _ = self.reward.waitTimeDict.get(edge, (1, 1)) # Pega o tempo de espera da aresta, se não existir, usa (1, 1) como padrão
        delay = self.dynamicDelays.get(edge, 0) # Tempo de espera dinâmico (se houver)
        total_time = wait_time + delay # Tempo total da aresta

        self.estimated_time_so_far += total_time # Acumula o tempo estimado

        print(f"Current state: {self.state}, Target: {self.target}, Estimated time so far: {self.estimated_time_so_far}, Delay: {delay}, Total time: {total_time}")

        # Usa a rewardClass e stopClass para calcular recompensa e término do episódio
        reward = self.reward.getReward(
            self.state, previousState, action, self.target, self.network, 
            self.estimated_time_so_far, self.max_expected_time, delay
        )

        terminated = self.stop.isTerminated(self.state, previousState, action, self.target, self.network)

        print("Reward on Step:",reward, "Terminated:",terminated)

        obs = (self.state, self.target)

        # Retorna: observação, recompensa, se o episódio terminou, se o episódio foi truncado (False), e um dicionário com metadados (count de passos)
        return obs, reward, terminated, False, {"count" : self.count}

    def generate_random_delay(self, start, target):
        try:
            # Encontra o caminho mais curto entre o start e o target
            shortest_path = nx.shortest_path(self.network, source=start, target=target, weight='weight')
            path_edges = list(zip(shortest_path, shortest_path[1:])) # Cria uma lista de arestas do caminho mais curto
            total_time = 0

            if not path_edges:
                return  # Sem arestas para atrasar

            # Calcula tempo total das arestas do caminho
            for a, b in path_edges:
                a, b = str(a), str(b)
                edge_key = (min(a, b), max(a, b)) # Ordena os nós da aresta para evitar duplicação
                if edge_key in self.reward.waitTimeDict:
                    edge_time = self.reward.waitTimeDict[edge_key][0]
                    total_time += edge_time # Soma o tempo de espera da aresta
                else:
                    print(f"[AVISO] Aresta {edge_key} não está no waitTimeDict!")

            average_time = total_time / len(path_edges)
            print(f"Média de tempo das arestas do caminho ótimo: {average_time}")

            # Escolhe uma das arestas do caminho para atrasar
            delay_u, delay_v = random.choice(path_edges)
            delay_u, delay_v = str(delay_u), str(delay_v) # Ordena os nós da aresta para evitar duplicação
            delay_edge_key = (min(delay_u, delay_v), max(delay_u, delay_v)) # Aresta escolhida para atraso

            if delay_edge_key in self.reward.waitTimeDict: # Verifica se a aresta escolhida está no waitTimeDict
                delay = average_time * 5  # Simula congestionamento pesado
                self.dynamicDelays = {
                    delay_edge_key: delay # Aresta escolhida para atraso com o tempo de atraso aplicado
                }
                print(f"Aresta atrasada: {delay_edge_key}, atraso aplicado: {delay}")
            else:
                print(f"[ERRO] Aresta escolhida para atraso {delay_edge_key} não está no waitTimeDict.")
                self.dynamicDelays = {}

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.dynamicDelays = {}

        
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
        with open('./output/combined_sum_amount.pkl', 'rb') as f: # Carrega o dicionário de tempos de espera e quantidades de viagens do DataSet
            self.waitTimeDict = pickle.load(f) # waitTimeDict[(u, v)] = (tempo de espera, quantidade)

    # A recompensa é calculada com base no tempo total e na quantidade de viagens
    def getReward(self, state, previousState, action, target, graph, estimated_time_so_far, max_expected_time, delay):
        
        totalTime = self.waitTimeDict[(previousState, state)][0]
        amount = self.waitTimeDict[(previousState, state)][1]

        # reward = 0 # A recompensa padrão é negativa: -totalTime / amount, incentivando caminhos com menor tempo médio

        # Recompensa negativa pelo tempo gasto
        # A recompensa é negativa, pois o agente deve minimizar o tempo gasto
        # Isso casa perfeitamente com algoritmos como Q-learning ou DQN, que maximizam retorno acumulado
        reward = - (totalTime + delay)

        # Se o agente chega no destino, dá um bônus proporcional ao tempo estimado e ao tempo máximo esperado
        if state == target:
            delay_ratio = estimated_time_so_far / max_expected_time if max_expected_time > 0 else 1
            if delay_ratio <= 1.2:
                reward += 500000 # Grandesa escolhida para ter um impacto significativo na recompensa total
            elif delay_ratio <= 1.5:
                reward += 250000
            else:
                reward -= 500000 # Penaliza se o tempo estimado for muito maior que o esperado
            print(f"Reward: {reward}, Estimated time so far: {estimated_time_so_far}, Max expected time: {max_expected_time}")
            print("delay_ratio: ", delay_ratio)
        
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
        max_steps = 5000  # Limite (generico) de passos por episódio, valor pode mudar

       #  rewardsPerStep = []  # Array para guardar a recompensa total de cada passo

        rewardsPerEpisode = np.zeros(episodes)


        try:
            if(is_training): # Se for treinamento, começamos com Q-table vazia
                q = {}
            else: # Se não for treinamento, carregamos a Q-table de um arquivo
                f = open('q.pkl', 'rb')
                q = pickle.load(f)
                f.close()
                
            stepsPerEpisode = np.zeros(episodes) # Array para guardar quantos passos foram dados em cada episódio
        
            for i in range(episodes): # Inicia o laço de episódios de treino/teste
                
                # Reinicia o ambiente (env.reset()) e recebe o estado inicial (posição atual + alvo)
                stepCount = 0 # contador de passos do episódio atual
                currentEpisodeReward = 0 # Variável para acumular a recompensa do episódio atual
                state = env.reset()[0]
                start = state[0]
                target = state[1]
                terminated = False # Terminated marca se o episódio terminou (atingiu o alvo)

                while(not terminated and stepCount < max_steps): # Enquanto o episódio não terminar seguir

                    neighbors = list(env.network.neighbors(state[0]))
                    if not neighbors:  # Se não houver vizinhos, termina o episódio
                        terminated = True
                        continue

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

                    currentEpisodeReward += reward # Acumula a recompensa do episódio atual

                    # rewardsPerStep.append(reward) # Acumula a recompensa total de cada passo

                    rewardsPerEpisode[i] = currentEpisodeReward
                    
                    # Mesma lógica que antes — se o novo estado ainda não está na Q-table, criamos entradas
                    if new_state not in q:
                        q[new_state] = {}
                        for act in range(len(list(env.network.neighbors(new_state[0])))):
                            q[new_state][act] = 0

                    # Atualização da Q-table (apenas se for treino)
                    # Essa é a equação padrão do Q-learning 
                    if is_training:

                        # A Q-table é atualizada com base na recompensa recebida e no valor máximo da próxima ação
                        qValues = [q[new_state][act] for act in q[new_state]] if q[new_state] else [0]
                        # qValues = [ q[new_state][act] for act in range(len(list(env.network.neighbors(new_state[0])))) ]
                        
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
                
                # rewardsPerStep[i] = currentEpisodeReward
                # Reduz o epsilon ao longo dos episódios para explorar menos e explorar mais com o tempo
                epsilon = max(epsilon - 1/(episodes), 0.01) # Nunca deixa epsilon menor que 0.01 (exploração mínima)
                if time.time() - startTime > 60 * 60 * 2: # 2 hours
                    break

            env.close() # Fecha o ambiente

        finally:
            # Calcula a média móvel dos passos por episódio (últimos 100 episódios)
            # Plota esse gráfico e salva
            print("np.zeros(i): ", np.zeros(i))
            sumSteps = np.zeros(i)
            for t in range(i):
                sumSteps[t] = np.mean(stepsPerEpisode[max(0, t-100):(t+1)]) # Média móvel dos passos por episódio

            plt.plot(sumSteps) # Plota a média móvel dos passos por episódio
            plt.savefig(f'q{env.initial}-{env.target}-{alpha}-{gamma}-{epsilon}.png')

            # Plota a RECOMPENSA total por passo
            #window_size = 100
            #moving_avg = [np.mean(rewardsPerStep[max(0, i-window_size):i+1]) for i in range(len(rewardsPerStep))] # Média móvel da recompensa total por passo
            
            # Plota a RECOMPENSA total por episódio (média móvel)
            window_size = 100
            moving_avg = [np.mean(rewardsPerEpisode[max(0, i-window_size):i+1]) for i in range(len(rewardsPerEpisode))]

            plt.figure(figsize=(12,6))
            plt.plot(moving_avg)
            plt.title("Recompensa por episódio (média móvel)")
            plt.xlabel("Episódios")
            plt.ylabel("Recompensa média")
            plt.grid(True)
            plt.savefig(f'rewards-smooth-{env.initial}-{env.target}-{alpha}-{gamma}-{epsilon}.png')

            # np.save(f'steps_{env.initial}_{env.target}.npy', stepsPerEpisode) # Salva o número de passos por episódio em um arquivo .npy
            # Se for treinamento, salva a Q-table em um arquivo
            if is_training:
                # Save Q Table
                f = open(f'q{env.initial}-{env.target}-{alpha}-{gamma}-{epsilon}.pkl',"wb")
                pickle.dump(q, f)
                f.close()


    with open('./sunt/graph_designer/graph_gtfs.gpickle', 'rb') as f: # Carrega o grafo salvo
        G = pickle.load(f)

    
    env = GraphExplorationEnv(G, 9) # Cria o ambiente com o grafo carregado e 9 ações possíveis (número máximo de vizinhos de um nó)

    run_q(30000, env, 0.9, 0.9, 0.2, is_training=True) # Executa run_q(...) com 20000 episódios e hiperparâmetros definidos (alpha=0.9, gamma=0.9, epsilon=0.2).

    #run_q(1, env, alpha=0.9, gamma=0.9, epsilon=0.0, is_training=False) # Teste da política aprendida
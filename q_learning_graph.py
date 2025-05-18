import numpy as np
import random

# Lê o grafo do arquivo
def carregar_grafo(caminho_arquivo):
    with open(caminho_arquivo, 'r') as f:
        linhas = f.read().splitlines()

    estado_inicial = linhas[0].split(':')[1].strip()
    estado_final = linhas[1].split(':')[1].strip()

    arestas = linhas[2:]

    grafo = {}
    for linha in arestas:
        origem, destino, custo = linha.split()
        custo = int(custo)

        if origem not in grafo:
            grafo[origem] = []
        grafo[origem].append((destino, custo))

    return grafo, estado_inicial, estado_final

# Cria o dicionário de estados e índice reverso
def mapear_estados(grafo):
    estados = set(grafo.keys())
    for vizinhos in grafo.values():
        for destino, _ in vizinhos:
            estados.add(destino)
    estados = sorted(list(estados))
    estado_para_indice = {estado: i for i, estado in enumerate(estados)}
    indice_para_estado = {i: estado for estado, i in estado_para_indice.items()}
    return estado_para_indice, indice_para_estado

# Inicializa Q e R
def inicializar_matrizes(grafo, estado_para_indice, estado_final):
    n = len(estado_para_indice)
    R = np.full((n, n), -1)
    Q = np.zeros((n, n))

    for origem, vizinhos in grafo.items():
        for destino, custo in vizinhos:
            i = estado_para_indice[origem]
            j = estado_para_indice[destino]
            if destino == estado_final:
                R[i][j] = 100  # Recompensa máxima ao chegar no destino
            else:
                R[i][j] = 0  # Pode ajustar com base no custo se quiser penalizar

    return Q, R

# Q-Learning
def treinar(Q, R, gamma=0.8, episodios=1000):
    n = Q.shape[0]

    for _ in range(episodios):
        estado_atual = random.randint(0, n - 1)
        while True:
            acoes_possiveis = np.where(R[estado_atual] >= 0)[0]
            if len(acoes_possiveis) == 0:
                break
            acao = random.choice(acoes_possiveis)
            max_q_proximo = np.max(Q[acao])
            Q[estado_atual][acao] = R[estado_atual][acao] + gamma * max_q_proximo
            estado_atual = acao
            if R[estado_atual][acao] == 100:
                break
    return Q

# Encontra o caminho ótimo
def encontrar_caminho(Q, estado_para_indice, indice_para_estado, estado_inicial, estado_final):
    caminho = [estado_inicial]
    estado_atual = estado_para_indice[estado_inicial]
    final = estado_para_indice[estado_final]

    while estado_atual != final:
        proximo = np.argmax(Q[estado_atual])
        estado_atual = proximo
        caminho.append(indice_para_estado[proximo])
        if len(caminho) > 100:
            break  # prevenção de loop infinito
    return caminho

# Execução principal
if __name__ == "__main__":
    grafo, estado_inicial, estado_final = carregar_grafo("graph_data.txt")
    estado_para_indice, indice_para_estado = mapear_estados(grafo)
    Q, R = inicializar_matrizes(grafo, estado_para_indice, estado_final)
    Q = treinar(Q, R)

    caminho = encontrar_caminho(Q, estado_para_indice, indice_para_estado, estado_inicial, estado_final)

    print("Caminho ótimo aprendido:")
    print(" -> ".join(caminho))

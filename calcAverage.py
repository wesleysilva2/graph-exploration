import os
import pandas as pd # leitura e manipulação dos dados, especialmente o .parquet.
import pickle # salvar objetos Python em arquivos binários (tipo dicionários).
import re # regex, pra identificar nomes de pastas com datas (YYYY-MM-DD).
import pyarrow.parquet as pq

"""
Esse código tem como objetivo processar arquivos .parquet de viagens de ônibus, calcular o tempo médio entre paradas (stop_id -> stop_id) 
para cada dia disponível, e salvar os resultados individualmente e combinados em arquivos .pkl.

"""


def process_date_folder(date_str: str) -> dict: # Recebe uma string de data no formato "YYYY-MM-DD"
    date = date_str.split("-") # Divide a string em partes (ano, mês, dia)
    file_path = f"./sunt/{date_str}/output/trips_time-series_{date[2]}-{date[1]}-{date[0]}_{date[2]}-{date[1]}-{date[0]}.parquet"
    if not os.path.exists(file_path): # Tenta encontrar o arquivo .parquet correspondente a essa data 
        print(f"File not found for date: {file_path}")
        return None
    
    df = pd.read_parquet(file_path) # Lê o .parquet com pandas
    # df  = pq.read_table(file_path)
    time_per_stop = {}  # array where [total time, number of trips]
    
    print(f"Processing {date_str}...")
    i = 0
    # Loop por todas as linhas do DataFrame:
    for index, row in df.iterrows():
        if index == 0: # Se for a primeira linha, não há linha anterior para comparar
            continue

        previous_row = df.loc[index-1] # Pega a linha anterior
    
        if previous_row["linha_atend"] != row["linha_atend"]: # Se a linha (linha_atend) anterior não for a mesma que a atual, pula 
            continue

        stop_pair = (previous_row["stop_id"], row["stop_id"]) # Define o par de paradas (parada anterior → parada atual)

        if stop_pair not in time_per_stop:
            time_per_stop[stop_pair] = [0, 0] # Inicializa o tempo total e o contador de viagens para esse par de paradas

        # Soma o tempo_total gasto nesse trecho e incrementa o contador de vezes que ele ocorreu
        time_per_stop[stop_pair][0] += row["tempo_total"]
        time_per_stop[stop_pair][1] += 1

        if index / len(df.index) * 100 > i: # Mostra o progresso em porcentagem no terminal
            i += 1
            print(f"\r{i}% of {len(df.index)} rows for {date_str}", end='')
    
    return time_per_stop

# Lista todas as pastas em ./sunt/ que representam datas no formato YYYY-MM-DD
base_path = "./sunt"
# print(f"\n📁 Verificando pastas em: {os.path.abspath(base_path)}")
# print("📂 Conteúdo da pasta sunt:", os.listdir(base_path) if os.path.exists(base_path) else "❌ Pasta não encontrada!")

date_folders = [f for f in os.listdir(base_path) 
                if os.path.isdir(os.path.join(base_path, f)) # Verifica se é uma pasta
                and re.match(r'\d{4}-\d{2}-\d{2}', f) # Verifica se o nome da pasta está no formato YYYY-MM-DD
                ]
date_folders.sort()

# print("📆 Pastas identificadas como datas:", date_folders)

# Process all dates and combine results
combined_averages = {}


for index, date_folder in enumerate(date_folders): # Itera por cada pasta de data, processando o .parquet associado
    print("Ta enterando ???")
    print(f"{index}/{len(date_folders)}")
    result = process_date_folder(date_folder) # Chama a função process_date_folder para processar os dados

    if result == None:
        continue
    
    with open(f"./output/averages_{date_folder}.pkl", "wb") as f: # Se houver resultado, salva em ./output/averages_YYYY-MM-DD.pkl
        pickle.dump(result, f)
    print(f"\nSaved results for {date_folder}")
    
    print(f"Combining {date_folder}...", end="\n\n")
    
    # Junta os tempos totais e contagens por par de parada em um grande dicionário combined_averages
    for stop_pair, (total_time, count) in result.items():
        if stop_pair not in combined_averages:
            combined_averages[stop_pair] = [0, 0]
        combined_averages[stop_pair][0] += total_time
        combined_averages[stop_pair][1] += count

print(f"Saving combined averages...")
# Salva o dicionário com as somas e contagens combinadas de todas as datas em ./output/combined_sum_amount.pkl
with open(f"./output/combined_sum_amount.pkl", "wb") as f:
    pickle.dump(combined_averages, f)

# Converte o dicionário de [soma, contagem] para uma média direta dividindo a soma pela contagem
# e salva em ./output/combined_averages.pkl 
final_averages = {
    stop_pair: total_time / count 
    for stop_pair, (total_time, count) in combined_averages.items()
}

final_averages
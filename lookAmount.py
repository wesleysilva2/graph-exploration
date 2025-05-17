import pickle

with open('./output/combined_sum_amount.pkl', 'rb') as f:
    data = pickle.load(f)

print("Total de pares:", len(data))
print("Exemplo:", list(data.items())[:5])

"""
Exemplo de saída:
(('44782944', '44784618'), [309447.88, 3216])
O trajeto de '44782944' para '44784618' foi feito 3.216 vezes, com um tempo acumulado de 309.447 segundos.
O tempo médio para esse trajeto é de 309.447 / 3.216 = 96.2 segundos.
"""

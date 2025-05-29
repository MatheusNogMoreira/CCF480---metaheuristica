
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import seaborn as sns

def func_objetivo2(x):
    x1, x2, x3, x4 = x
    termo1 = 100 * (x1**2 - x2)**2
    termo2 = (x1 - 1)**2
    termo3 = (x3 - 1)**2
    termo4 = 90 * (x3**2 - x4)**2
    termo5 = 10.1 * ((x2 - 1)**2 + (x4 - 1)**2)
    termo6 = 19.8 * (x2 - 1) * (x4 - 1)
    return termo1 + termo2 + termo3 + termo4 + termo5 + termo6

def gerar_vizinho(x, k, intervalo):
    vizinho = x.copy()
    for i in range(len(x)):
        delta = k * (intervalo[i][1] - intervalo[i][0]) * 0.05
        vizinho[i] += np.random.uniform(-delta, delta)
        vizinho[i] = np.clip(vizinho[i], intervalo[i][0], intervalo[i][1])
    return vizinho

def melhor_vizinho(x, k_max, intervalo):
    melhor = x.copy()
    melhor_val = func_objetivo2(x)
    for k in range(1, k_max + 1):
        x_viz = gerar_vizinho(x, k, intervalo)
        val = func_objetivo2(x_viz)
        if val < melhor_val:
            melhor, melhor_val = x_viz, val
    return melhor

def VNS(intervalo, iter_max=1000, k_max=3):
    x = [np.random.uniform(low, high) for low, high in intervalo]
    fx = func_objetivo2(x)
    
    for _ in range(iter_max):
        k = 1
        while k <= k_max:
            x_viz = gerar_vizinho(x, k, intervalo)
            x_viz = melhor_vizinho(x_viz, k_max, intervalo)
            f_viz = func_objetivo2(x_viz)
            if f_viz < fx:
                x, fx = x_viz, f_viz
                k = 1
            else:
                k += 1
    return x, fx

def SA(intervalo, iter_max=1000, T0=1000, alpha=0.95):
    x = [np.random.uniform(low, high) for low, high in intervalo]
    fx = func_objetivo2(x)
    T = T0
    
    for _ in range(iter_max):
        x_viz = gerar_vizinho(x, 1, intervalo)
        f_viz = func_objetivo2(x_viz)
        delta = f_viz - fx
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            x, fx = x_viz, f_viz
        T *= alpha
    return x, fx

def executar_experimentos(algoritmo, intervalo, label):
    resultados = []
    melhores_solucoes = []
    for _ in range(30):
        x, fx = algoritmo(intervalo)
        resultados.append(fx)
        melhores_solucoes.append(x)
    return resultados, melhores_solucoes

def estatisticas(nome_alg, resultados):
    print(f"\n{nome_alg}:")
    print(f"  Mínimo: {np.min(resultados):.4f}")
    print(f"  Máximo: {np.max(resultados):.4f}")
    print(f"  Média: {np.mean(resultados):.4f}")
    print(f"  Desvio Padrão: {np.std(resultados):.4f}")

def gerar_boxplot(dados_dict, titulo):
    sns.boxplot(data=list(dados_dict.values()))
    plt.xticks(range(len(dados_dict)), list(dados_dict.keys()))
    plt.title(titulo)
    plt.ylabel("Valor da função objetivo")
    plt.grid(True)
    plt.show()

def mostrar_melhor(label, resultados, solucoes):
    idx = np.argmin(resultados)
    print(f"\nMelhor solução - {label}")
    print(f"x = {np.round(solucoes[idx], 4)}")
    print(f"f(x) = {resultados[idx]:.4f}")

# Intervalo (c)
intervalo_c = [(-10, 10)] * 4
# Intervalo (d)
intervalo_d = [(-2, 2)] * 4

# Executa para intervalo (c)
vns_c, sol_vns_c = executar_experimentos(VNS, intervalo_c, "VNS")
sa_c, sol_sa_c = executar_experimentos(SA, intervalo_c, "SA")
estatisticas("VNS (c)", vns_c)
estatisticas("SA (c)", sa_c)
gerar_boxplot({"VNS": vns_c, "SA": sa_c}, "Boxplot - Intervalo (c)")

# Executa para intervalo (d)
vns_d, sol_vns_d = executar_experimentos(VNS, intervalo_d, "VNS")
sa_d, sol_sa_d = executar_experimentos(SA, intervalo_d, "SA")
estatisticas("VNS (d)", vns_d)
estatisticas("SA (d)", sa_d)
gerar_boxplot({"VNS": vns_d, "SA": sa_d}, "Boxplot - Intervalo (d)")

# Mostrar melhores soluções
mostrar_melhor("VNS (c)", vns_c, sol_vns_c)
mostrar_melhor("SA (c)", sa_c, sol_sa_c)
mostrar_melhor("VNS (d)", vns_d, sol_vns_d)
mostrar_melhor("SA (d)", sa_d, sol_sa_d)

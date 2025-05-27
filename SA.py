import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
INSTANCE_DIR = "instances"
INSTANCE_PREFIX = "ep"
N_INSTANCES = 6 #mudar aqui
N_EXECUCOES = 30
INITIAL_TEMP = 1000
COOLING_RATE = 0.995
MAX_ITER = 5000

# --- LEITURA DOS DADOS ---
def load_instance(filename):
    try:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
        
        if not lines:
            raise ValueError(f"Arquivo vazio: {filename}")
        
        # Primeira linha: W ? ?
        first_line = lines[0].split()
        W = int(first_line[0])
        
        # Sabores (t_i): linhas restantes
        t = []
        for line in lines[1:]:
            t.extend(map(int, line.split()))
        
        n = len(t)
        
        # Pesos (w_i): ASSUMINDO que são iguais a 1 (substitua conforme necessário)
        w = np.ones(n)
        
        # Pares incompatíveis (I): ASSUMINDO vazio (substitua conforme necessário)
        I = []
        
        return W, w, t, I
    
    except Exception as e:
        print(f"Erro ao ler {filename}: {str(e)}")
        return None

# --- SIMULATED ANNEALING ---
def evaluate(solution, W, w, t, I):
    total_sabor = np.sum(t * solution)
    total_peso = np.sum(w * solution)
    penalty = 0
    
    if total_peso > W:
        penalty += (total_peso - W) * 1000
    
    for j, k in I:
        if solution[j] + solution[k] > 1:
            penalty += 1000
    
    return total_sabor - penalty

def initial_solution(n, W, w, I):
    solution = np.zeros(n)
    indices = list(range(n))
    random.shuffle(indices)
    
    for i in indices:
        if w[i] + np.sum(w * solution) <= W:
            compatible = True
            for j, k in I:
                if (i == j and solution[k] == 1) or (i == k and solution[j] == 1):
                    compatible = False
                    break
            if compatible:
                solution[i] = 1
    return solution

def get_neighbor(solution, W, w, I):
    n = len(solution)
    new_solution = solution.copy()
    i = random.randint(0, n-1)
    new_solution[i] = 1 - new_solution[i]
    
    while True:
        total_peso = np.sum(w * new_solution)
        invalid_pairs = [ (j,k) for j,k in I if new_solution[j] + new_solution[k] > 1 ]
        
        if total_peso <= W and not invalid_pairs:
            break
        
        idx = random.choice([i for i in range(n) if new_solution[i] == 1])
        new_solution[idx] = 0
    
    return new_solution

def simulated_annealing(W, w, t, I):
    n = len(t)
    current_sol = initial_solution(n, W, w, I)
    current_cost = evaluate(current_sol, W, w, t, I)
    best_sol = current_sol.copy()
    best_cost = current_cost
    temp = INITIAL_TEMP
    
    for _ in range(MAX_ITER):
        neighbor = get_neighbor(current_sol, W, w, I)
        neighbor_cost = evaluate(neighbor, W, w, t, I)
        delta = neighbor_cost - current_cost
        
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_sol, current_cost = neighbor, neighbor_cost
            
            if neighbor_cost > best_cost:
                best_sol, best_cost = neighbor, neighbor_cost
        
        temp *= COOLING_RATE
    
    return best_sol, best_cost

# --- EXECUÇÃO PRINCIPAL ---
def main():
    # Verifica se a pasta de instâncias existe
    if not os.path.exists(INSTANCE_DIR):
        print(f"Erro: Pasta '{INSTANCE_DIR}' não encontrada!")
        return
    
    results = {}
    
    for i in range(6, N_INSTANCES + 1):
        instance_name = f"{INSTANCE_PREFIX}{i:02d}"
        instance_path = os.path.join(INSTANCE_DIR, f"{instance_name}.dat")
        
        # Carrega a instância
        instance_data = load_instance(instance_path)
        if instance_data is None:
            continue
        
        W, w, t, I = instance_data
        instance_results = []
        
        print(f"\nExecutando {instance_name}...")
        for _ in tqdm(range(N_EXECUCOES), desc=instance_name):
            _, best_cost = simulated_annealing(W, w, t, I)
            instance_results.append(best_cost)
        
        results[instance_name] = instance_results
    
    if not results:
        print("Nenhum resultado foi gerado. Verifique os arquivos de instância.")
        return
    
    # --- ANÁLISE ESTATÍSTICA ---
    stats = pd.DataFrame.from_dict(results, orient='index')
    stats['Média'] = stats.mean(axis=1)
    stats['Mínimo'] = stats.min(axis=1)
    stats['Máximo'] = stats.max(axis=1)
    stats['Desvio Padrão'] = stats.std(axis=1)
    
    print("\n=== Estatísticas ===")
    print(stats[['Média', 'Mínimo', 'Máximo', 'Desvio Padrão']])
    
    # --- BOXPLOT ---
    plt.figure(figsize=(12, 6))
    plot_data = pd.DataFrame.from_dict(results, orient='index').T
    plot_data.boxplot()
    plt.title("Distribuição do Sabor Total por Instância")
    plt.ylabel("Sabor Total")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("boxplot.png")
    plt.show()
    
    # --- SALVAR RESULTADOS COMPLETOS ---
    with open("resultados_completos.txt", 'w') as f:
        for instance in results:
            f.write(f"=== {instance} ===\n")
            f.write(f"Melhor resultado: {max(results[instance])}\n")
            f.write(f"Menor resultado: {min(results[instance])}\n")
            f.write(f"Média: {np.mean(results[instance]):.2f}\n")
            f.write(f"Desvio padrão: {np.std(results[instance]):.2f}\n")
            f.write(f"Todos os resultados: {sorted(results[instance], reverse=True)}\n\n")
            
        # Resumo geral
        f.write("\n=== RESUMO GERAL ===\n")
        f.write(stats[['Média', 'Mínimo', 'Máximo', 'Desvio Padrão']].to_string())

if __name__ == "__main__":
    main()
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
INSTANCE_DIR = "instances"
INSTANCE_PREFIX = "ep"
N_INSTANCES = 10
N_EXECUCOES = 30
INITIAL_TEMP = 1000
COOLING_RATE = 0.995
MAX_ITER = 5000

# Melhores valores conhecidos (BKV)
BKVs = {
    'ep01': 2118, 'ep02': 1378, 'ep03': 2850, 'ep04': 2730, 'ep05': 2624,
    'ep06': 4690, 'ep07': 4440, 'ep08': 5020, 'ep09': 4568, 'ep10': 4390
}

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
        w = np.ones(n)  # Assumindo pesos unitários (substitua se necessário)
        I = []          # Assumindo sem incompatibilidades (substitua se necessário)
        
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
    start_time = time.time()
    
    initial_sol = initial_solution(n, W, w, I)
    initial_cost = evaluate(initial_sol, W, w, t, I)
    
    current_sol = initial_sol.copy()
    current_cost = initial_cost
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
    
    exec_time = time.time() - start_time
    return {
        'initial_sol': initial_sol,
        'initial_cost': initial_cost,
        'final_sol': best_sol,
        'final_cost': best_cost,
        'time': exec_time
    }

# --- EXECUÇÃO PRINCIPAL ---
def main():
    if not os.path.exists(INSTANCE_DIR):
        print(f"Erro: Pasta '{INSTANCE_DIR}' não encontrada!")
        return
    
    results = {}
    detailed_stats = []
    all_executions = []

    for i in range(1, N_INSTANCES + 1):
        instance_name = f"{INSTANCE_PREFIX}{i:02d}"
        instance_path = os.path.join(INSTANCE_DIR, f"{instance_name}.dat")
        
        instance_data = load_instance(instance_path)
        if instance_data is None:
            continue
        
        W, w, t, I = instance_data
        instance_results = []
        best_exec = None
        
        print(f"\nExecutando {instance_name}...")
        for _ in tqdm(range(N_EXECUCOES), desc=instance_name):
            sa_result = simulated_annealing(W, w, t, I)
            instance_results.append(sa_result['final_cost'])
            all_executions.append({
                'Instância': instance_name,
                'Execução': _ + 1,
                'Custo_Final': sa_result['final_cost']
            })
            
            if best_exec is None or sa_result['final_cost'] > best_exec['final_cost']:
                best_exec = sa_result
        
        BKV = BKVs.get(instance_name, 0)
        delta_initial = 100 * (best_exec['final_cost'] - best_exec['initial_cost']) / best_exec['initial_cost'] if best_exec['initial_cost'] != 0 else 0
        delta_optimal = 100 * (BKV - best_exec['final_cost']) / BKV if BKV != 0 else 0
        
        detailed_stats.append({
            'Instância': instance_name,
            'SI': best_exec['initial_cost'],
            'SF': best_exec['final_cost'],
            'Δ(SI→SF) (%)': delta_initial,
            'Δ(Ótimo) (%)': delta_optimal,
            'Tempo (s)': best_exec['time']
        })
        
        results[instance_name] = instance_results
    
    # --- ESTATÍSTICAS GERAIS ---
    if not results:
        print("Nenhum resultado foi gerado. Verifique os arquivos de instância.")
        return
    
    stats = pd.DataFrame.from_dict(results, orient='index')
    stats['Média'] = stats.mean(axis=1)
    stats['Mínimo'] = stats.min(axis=1)
    stats['Máximo'] = stats.max(axis=1)
    stats['Desvio Padrão'] = stats.std(axis=1)
    
    print("\n=== Estatísticas das 30 Execuções ===")
    print(stats[['Média', 'Mínimo', 'Máximo', 'Desvio Padrão']])
    
    # --- BOXPLOT ---
    plt.figure(figsize=(12, 6))
    pd.DataFrame.from_dict(results, orient='index').T.boxplot()
    plt.title("Distribuição do Sabor Total por Instância (30 Execuções)")
    plt.ylabel("Sabor Total")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("boxplot_30_execucoes.png")
    plt.show()
    
    # --- ANÁLISE DETALHADA (MELHOR EXECUÇÃO) ---
    print("\n=== Análise Detalhada (Melhor Execução por Instância) ===")
    detailed_df = pd.DataFrame(detailed_stats)
    print(detailed_df)
    
    # --- SALVAMENTO DE DADOS ---
    detailed_df.to_csv("detalhes_melhores_execucoes.csv", index=False)
    pd.DataFrame(all_executions).to_csv("todas_execucoes.csv", index=False)
    
    # Salva as melhores soluções encontradas
    with open("melhores_solucoes.txt", 'w') as f:
        for instance in results:
            f.write(f"{instance}: {max(results[instance])}\n")

if __name__ == "__main__":
    main()
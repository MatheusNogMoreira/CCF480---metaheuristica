import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Função para ler o arquivo .dat
def read_instance(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse primeira linha
    n, W, _ = map(int, lines[0].split())

    # Pega pesos e sabores
    weights = []
    flavors = []

    idx = 1
    while len(weights) < n:
        weights.extend(map(int, lines[idx].split()))
        idx += 1

    while len(flavors) < n:
        flavors.extend(map(int, lines[idx].split()))
        idx += 1

    # Pega pares de incompatibilidade
    incompatibilities = set()
    for line in lines[idx:]:
        parts = line.strip().split()
        if len(parts) == 2:
            j, k = map(int, parts)
            incompatibilities.add((j - 1, k - 1))  # índice base 0
            incompatibilities.add((k - 1, j - 1))  # bidirecional

    return n, W, weights, flavors, incompatibilities

# Função para avaliar uma solução
def evaluate(solution, W, weights, flavors, incompatibilities):
    total_weight = 0
    total_flavor = 0

    selected = [i for i, x in enumerate(solution) if x == 1]

    for i in selected:
        for j in selected:
            if i != j and (i, j) in incompatibilities:
                return -1

    for i in selected:
        total_weight += weights[i]
        total_flavor += flavors[i]

    if total_weight > W:
        return -1  

    return total_flavor

# Geração de solução inicial viável (gulosa aleatória)
def initial_solution(n, W, weights, flavors, incompatibilities):
    solution = [0] * n
    indices = list(range(n))
    random.shuffle(indices)

    total_weight = 0
    selected_set = set()

    for i in indices:
        # checa se adicionar i quebra alguma incompatibilidade
        compatible = True
        for j in selected_set:
            if (i, j) in incompatibilities:
                compatible = False
                break

        if compatible and total_weight + weights[i] <= W:
            solution[i] = 1
            total_weight += weights[i]
            selected_set.add(i)

    return solution

def neighbor(solution):
    n = len(solution)
    new_solution = solution.copy()

    if random.random() < 0.5:
        i = random.randint(0, n - 1)
        new_solution[i] = 1 - new_solution[i]
    else:
        ones = [i for i, x in enumerate(solution) if x == 1]
        zeros = [i for i, x in enumerate(solution) if x == 0]

        if ones and zeros:
            i = random.choice(ones)
            j = random.choice(zeros)
            new_solution[i] = 0
            new_solution[j] = 1

    return new_solution

# SA
def simulated_annealing(n, W, weights, flavors, incompatibilities,
                        initial_temp=1000.0, final_temp=0.1, alpha=0.98, iter_per_temp=100):
    
    current_solution = initial_solution(n, W, weights, flavors, incompatibilities)
    current_value = evaluate(current_solution, W, weights, flavors, incompatibilities)
    best_solution = current_solution.copy()
    best_value = current_value

    T = initial_temp

    while T > final_temp:
        for _ in range(iter_per_temp):
            new_solution = neighbor(current_solution)
            new_value = evaluate(new_solution, W, weights, flavors, incompatibilities)

            if new_value == -1:
                continue  # solução inviável

            delta = new_value - current_value

            if delta > 0 or random.random() < math.exp(delta / T):
                current_solution = new_solution
                current_value = new_value

                if current_value > best_value:
                    best_solution = current_solution.copy()
                    best_value = current_value

        T *= alpha

    return best_solution, best_value

# Rodar 30 execuções para uma instância
def run_experiment(instance_filename, output_prefix):
    n, W, weights, flavors, incompatibilities = read_instance(instance_filename)

    print(f"Instance: {instance_filename}")
    print(f"n = {n}, W = {W}, #incompatibilities = {len(incompatibilities)//2}")

    results = []
    best_overall_solution = None
    best_overall_value = -1

    for run in range(30):
        best_solution, best_value = simulated_annealing(n, W, weights, flavors, incompatibilities)

        print(f"Run {run+1}: flavor = {best_value}")
        results.append(best_value)

        if best_value > best_overall_value:
            best_overall_value = best_value
            best_overall_solution = best_solution.copy()

    # Estatísticas
    mean_val = np.mean(results)
    min_val = np.min(results)
    max_val = np.max(results)
    std_val = np.std(results)

    print("\n=== Statistics ===")
    print(f"Mean: {mean_val:.2f}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Std: {std_val:.2f}")

    # Gravar boxplot
    plt.figure(figsize=(8,6))
    plt.boxplot(results)
    plt.title(f'Boxplot - {instance_filename}')
    plt.ylabel('Total Flavor')
    plt.grid(True)
    plt.savefig(f'{output_prefix}_boxplot.png')
    plt.close()

    # Gravar melhor solução em arquivo
    selected = [i + 1 for i, x in enumerate(best_overall_solution) if x == 1]
    with open(f'{output_prefix}_best_solution.txt', 'w') as f:
        f.write(f"Best flavor: {best_overall_value}\n")
        f.write("Selected ingredients:\n")
        f.write(" ".join(map(str, selected)) + "\n")

    # Gravar tabela de resultados
    with open(f'{output_prefix}_results.txt', 'w') as f:
        f.write(f"Mean: {mean_val:.2f}\n")
        f.write(f"Min: {min_val}\n")
        f.write(f"Max: {max_val}\n")
        f.write(f"Std: {std_val:.2f}\n")
        f.write("\nAll results:\n")
        for val in results:
            f.write(f"{val}\n")

# Main
if __name__ == "__main__":
    instance_filename = "instances/ep05.dat"
    output_prefix = "ep05"

    run_experiment(instance_filename, output_prefix)

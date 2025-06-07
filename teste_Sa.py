import random
import math
import matplotlib.pyplot as plt
import numpy as np

def read_instance(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Remove linhas em branco e espaços
    lines = [line.strip() for line in lines]

    # 1ª linha: n, incompat_count, W
    n, incompat_count, W = map(int, lines[0].split())

    # Ler sabores: podem estar em várias linhas até totalizar n valores
    flavors = []
    idx = 2  # começa após a linha em branco da linha 1
    while len(flavors) < n:
        if lines[idx] != '':
            flavors.extend(map(int, lines[idx].split()))
        idx += 1

    # Ler pesos: mesmas regras, logo após sabores
    weights = []
    while len(weights) < n:
        if lines[idx] != '':
            weights.extend(map(int, lines[idx].split()))
        idx += 1

    # Ler incompatibilidades
    incompatibilities = set()
    for line in lines[idx:idx+incompat_count]:
        if line != '':
            j, k = map(int, line.split())
            incompatibilities.add((j - 1, k - 1))
            incompatibilities.add((k - 1, j - 1))

    return n, W, weights, flavors, incompatibilities

def evaluate(solution, W, weights, flavors, incompatibilities, penalty_weight=10, penalty_incompat=50):
    total_weight = 0
    total_flavor = 0
    penalty = 0
    selected = [i for i, x in enumerate(solution) if x]

    for i in selected:
        total_weight += weights[i]
        total_flavor += flavors[i]

    if total_weight > W:
        penalty += penalty_weight * (total_weight - W)

    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if (selected[i], selected[j]) in incompatibilities:
                penalty += penalty_incompat

    return total_flavor - penalty

def initial_solution(n, W, weights, flavors, incompatibilities):
    solution = [0] * n
    indices = list(range(n))
    random.shuffle(indices)

    total_weight = 0
    selected_set = set()

    for i in indices:
        compatible = all((i, j) not in incompatibilities for j in selected_set)
        if compatible and total_weight + weights[i] <= W:
            solution[i] = 1
            total_weight += weights[i]
            selected_set.add(i)

    return solution

def neighbor(solution):
    n = len(solution)
    new_solution = solution.copy()

    op = random.choice(['flip', 'swap', 'add', 'remove'])

    if op == 'flip':
        i = random.randint(0, n - 1)
        new_solution[i] = 1 - new_solution[i]

    elif op == 'swap':
        ones = [i for i, x in enumerate(solution) if x == 1]
        zeros = [i for i, x in enumerate(solution) if x == 0]
        if ones and zeros:
            i = random.choice(ones)
            j = random.choice(zeros)
            new_solution[i] = 0
            new_solution[j] = 1

    elif op == 'add':
        zeros = [i for i, x in enumerate(solution) if x == 0]
        if zeros:
            i = random.choice(zeros)
            new_solution[i] = 1

    elif op == 'remove':
        ones = [i for i, x in enumerate(solution) if x == 1]
        if ones:
            i = random.choice(ones)
            new_solution[i] = 0

    return new_solution

def simulated_annealing(n, W, weights, flavors, incompatibilities,
                        initial_temp=1000.0, final_temp=0.1, alpha=0.95, iter_per_temp=300):
    
    current_solution = initial_solution(n, W, weights, flavors, incompatibilities)
    current_value = evaluate(current_solution, W, weights, flavors, incompatibilities)
    best_solution = current_solution.copy()
    best_value = current_value

    T = initial_temp
    stagnant = 0

    while T > final_temp:
        for _ in range(iter_per_temp):
            new_solution = neighbor(current_solution)
            new_value = evaluate(new_solution, W, weights, flavors, incompatibilities)

            delta = new_value - current_value

            if delta > 0 or random.random() < math.exp(delta / T):
                current_solution = new_solution
                current_value = new_value

                if current_value > best_value:
                    best_solution = current_solution.copy()
                    best_value = current_value
                    stagnant = 0
                else:
                    stagnant += 1

        if stagnant >= 500:
            # força uma reinicialização leve se estagnado
            current_solution = initial_solution(n, W, weights, flavors, incompatibilities)
            current_value = evaluate(current_solution, W, weights, flavors, incompatibilities)
            stagnant = 0

        T *= alpha

    return best_solution, best_value

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

    mean_val = np.mean(results)
    min_val = np.min(results)
    max_val = np.max(results)
    std_val = np.std(results)

    print("\n=== Statistics ===")
    print(f"Mean: {mean_val:.2f}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Std: {std_val:.2f}")

    plt.figure(figsize=(8,6))
    plt.boxplot(results)
    plt.title(f'Boxplot - {instance_filename}')
    plt.ylabel('Total Flavor')
    plt.grid(True)
    plt.savefig(f'{output_prefix}_boxplot.png')
    plt.close()

    selected = [i + 1 for i, x in enumerate(best_overall_solution) if x == 1]
    with open(f'{output_prefix}_best_solution.txt', 'w') as f:
        f.write(f"Best flavor: {best_overall_value}\n")
        f.write("Selected ingredients:\n")
        f.write(" ".join(map(str, selected)) + "\n")

    with open(f'{output_prefix}_results.txt', 'w') as f:
        f.write(f"Mean: {mean_val:.2f}\n")
        f.write(f"Min: {min_val}\n")
        f.write(f"Max: {max_val}\n")
        f.write(f"Std: {std_val:.2f}\n")
        f.write("\nAll results:\n")
        for val in results:
            f.write(f"{val}\n")

if __name__ == "__main__":
    instance_filename = "instances/ep04.dat"
    output_prefix = "ep04"
    run_experiment(instance_filename, output_prefix)

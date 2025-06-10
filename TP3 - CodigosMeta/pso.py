import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time

BKV = {
    "ep01": 2118,
    "ep02": 1378,
    "ep03": 2850,
    "ep04": 2730,
    "ep05": 2624,
    "ep06": 4690,
    "ep07": 4440,
    "ep08": 5020,
    "ep09": 4568,
    "ep10": 4390
}

# Função para ler o arquivo .dat (mantida igual)
def read_instance(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]

    n, incompat_count, W = map(int, lines[0].split())

    flavors = []
    idx = 2
    while len(flavors) < n:
        if lines[idx] != '':
            flavors.extend(map(int, lines[idx].split()))
        idx += 1

    weights = []
    while len(weights) < n:
        if lines[idx] != '':
            weights.extend(map(int, lines[idx].split()))
        idx += 1

    incompatibilities = set()
    for line in lines[idx:idx+incompat_count]:
        if line != '':
            j, k = map(int, line.split())
            incompatibilities.add((j - 1, k - 1))
            incompatibilities.add((k - 1, j - 1))

    return n, W, weights, flavors, incompatibilities

# Função de avaliação (mantida igual)
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

# Geração de solução inicial viável (modificada para PSO)
def initial_solution(n, W, weights, flavors, incompatibilities):
    solution = [0] * n
    indices = list(range(n))
    random.shuffle(indices)

    total_weight = 0
    selected_set = set()

    for i in indices:
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

# Classe para representar uma partícula no PSO
class Particle:
    def __init__(self, n, W, weights, flavors, incompatibilities):
        self.position = initial_solution(n, W, weights, flavors, incompatibilities)
        self.velocity = [random.uniform(-1, 1) for _ in range(n)]
        self.best_position = self.position.copy()
        self.best_value = evaluate(self.position, W, weights, flavors, incompatibilities)
        self.n = n
        
    def update_velocity(self, global_best_position, w=0.5, c1=1.0, c2=1.0):
        for i in range(self.n):
            r1 = random.random()
            r2 = random.random()
            
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best_position[i] - self.position[i])
            
            self.velocity[i] = w * self.velocity[i] + cognitive + social
            
            # Limitar velocidade para evitar explosão
            if self.velocity[i] > 1:
                self.velocity[i] = 1
            elif self.velocity[i] < -1:
                self.velocity[i] = -1
    
    def update_position(self, W, weights, flavors, incompatibilities):
        # Converter velocidade em probabilidade usando sigmoid
        for i in range(self.n):
            prob = 1 / (1 + math.exp(-self.velocity[i]))
            
            if random.random() < prob:
                self.position[i] = 1
            else:
                self.position[i] = 0
        
        # Reparar solução se for inviável
        self.repair_solution(W, weights, incompatibilities)
        
        # Avaliar nova posição
        current_value = evaluate(self.position, W, weights, flavors, incompatibilities)
        
        # Atualizar melhor posição pessoal
        if current_value > self.best_value:
            self.best_position = self.position.copy()
            self.best_value = current_value
            
        return current_value
    
    def repair_solution(self, W, weights, incompatibilities):
        # Verificar incompatibilidades
        selected = [i for i, x in enumerate(self.position) if x == 1]
        to_remove = set()
        
        # Identificar incompatibilidades
        for i in selected:
            for j in selected:
                if i != j and (i, j) in incompatibilities:
                    if weights[i] > weights[j]:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
        
        # Remover incompatibilidades
        for idx in to_remove:
            self.position[idx] = 0
        
        # Verificar capacidade
        total_weight = sum(weights[i] for i, x in enumerate(self.position) if x == 1)
        
        # Se exceder a capacidade, remover itens aleatoriamente até ficar viável
        while total_weight > W:
            selected = [i for i, x in enumerate(self.position) if x == 1]
            if not selected:
                break
                
            idx = random.choice(selected)
            self.position[idx] = 0
            total_weight -= weights[idx]

# Algoritmo PSO
def particle_swarm_optimization(n, W, weights, flavors, incompatibilities,
                              num_particles=30, max_iter=200, w=0.5, c1=1.0, c2=1.0):
    
    # Inicializar população de partículas
    particles = [Particle(n, W, weights, flavors, incompatibilities) for _ in range(num_particles)]
    
    # Encontrar melhor global inicial
    global_best_position = particles[0].position.copy()
    global_best_value = particles[0].best_value
    
    for particle in particles[1:]:
        if particle.best_value > global_best_value:
            global_best_position = particle.best_position.copy()
            global_best_value = particle.best_value
    
    initial_value = global_best_value
    
    # Executar iterações do PSO
    for _ in range(max_iter):
        for particle in particles:
            # Atualizar velocidade e posição
            particle.update_velocity(global_best_position, w, c1, c2)
            current_value = particle.update_position(W, weights, flavors, incompatibilities)
            
            # Atualizar melhor global
            if current_value > global_best_value:
                global_best_position = particle.position.copy()
                global_best_value = current_value
    
    return global_best_position, global_best_value, initial_value

# Rodar 30 execuções para uma instância (modificada para PSO)
def run_experiment(instance_filename, output_prefix):
    n, W, weights, flavors, incompatibilities = read_instance(instance_filename)

    print(f"Instance: {instance_filename}")
    print(f"n = {n}, W = {W}, #incompatibilities = {len(incompatibilities)//2}")

    results = []
    best_overall_solution = None
    best_overall_value = -1

    start_time = time.time()

    for run in range(30):
        best_solution, best_value, initial_value = particle_swarm_optimization(
            n, W, weights, flavors, incompatibilities
        )

        print(f"Run {run+1}: flavor = {best_value}")
        results.append(best_value)

        if best_value > best_overall_value:
            best_overall_value = best_value
            best_overall_solution = best_solution.copy()
        
    end_time = time.time()

    # Estatísticas (mantido igual)
    mean_val = np.mean(results)
    min_val = np.min(results)
    max_val = np.max(results)
    std_val = np.std(results)

    elapsed_time = end_time - start_time

    desvio_si_sf = 100*(initial_value - max_val) / initial_value if initial_value != 0 else 0
    desvio_sf_otimo = 100*(BKV[output_prefix] - max_val) / BKV[output_prefix] if BKV[output_prefix] != 0 else 0

    print("\n=== Statistics ===")
    print(f"Mean: {mean_val:.2f}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Std: {std_val:.2f}")

    print(f"BKV: {BKV[output_prefix]}")
    print(f"Initial value: {initial_value}")
    print(f"Final value: {max_val}")
    print(f"Desvio SI/SF: {desvio_si_sf:.2f}%")
    print(f"Desvio SF/Otimo: {desvio_sf_otimo:.2f}%")
    print(f"Time: {elapsed_time:.2f} seconds")

    # Gravar boxplot (mantido igual)
    plt.figure(figsize=(8,6))
    plt.boxplot(results)
    plt.title(f'Boxplot - {instance_filename}')
    plt.ylabel('Total Flavor')
    plt.grid(True)
    plt.savefig(f'{output_prefix}_boxplot.png')
    plt.close()

    # Gravar melhor solução em arquivo (mantido igual)
    selected = [i + 1 for i, x in enumerate(best_overall_solution) if x == 1]
    with open(f'{output_prefix}_best_solution.txt', 'w') as f:
        f.write(f"Best flavor: {best_overall_value}\n")
        f.write("Selected ingredients:\n")
        f.write(" ".join(map(str, selected)) + "\n\n")
        
        f.write(f"BKV: {BKV[output_prefix]}\n")
        f.write(f"Initial value: {initial_value}\n")
        f.write(f"Final value: {max_val}\n")
        f.write(f"Desvio SI/SF: {desvio_si_sf:.2f}%\n")
        f.write(f"Desvio SF/Ótimo: {desvio_sf_otimo:.2f}%\n")
        f.write(f"Time: {elapsed_time:.2f} seconds\n")

    # Gravar tabela de resultados (mantido igual)
    with open(f'{output_prefix}_results.txt', 'w') as f:
        f.write(f"Mean: {mean_val:.2f}\n")
        f.write(f"Min: {min_val}\n")
        f.write(f"Max: {max_val}\n")
        f.write(f"Std: {std_val:.2f}\n\n")
        
        f.write(f"BKV: {BKV[output_prefix]}\n")
        f.write(f"Initial value: {initial_value}\n")
        f.write(f"Final value: {max_val}\n")
        f.write(f"Desvio SI/SF: {desvio_si_sf:.2f}%\n")
        f.write(f"Desvio SF/Ótimo: {desvio_sf_otimo:.2f}%\n")
        f.write(f"Time: {elapsed_time:.2f} seconds\n\n")
        f.write("\nAll results:\n")
        for val in results:
            f.write(f"{val}\n")

# Main (mantido igual)
if __name__ == "__main__":
    for i in range(1, 11):
        filename = f"instances/ep{str(i).zfill(2)}.dat"
        output_prefix = f"ep{str(i).zfill(2)}"
        run_experiment(filename, output_prefix)
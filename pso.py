import numpy as np
import random
import time
from typing import List, Tuple
import matplotlib.pyplot as plt

class PSO:
    def __init__(
        self,
        n_ingredients: int,
        max_weight: int,
        weights: List[float],
        tastes: List[float],
        incompatibilities: List[Tuple[int, int]],
        n_particles: int = 30,
        max_iter: int = 100,
        inertia: float = 0.7,
        cognitive_weight: float = 1.5,
        social_weight: float = 1.5,
    ):
        self.n_ingredients = n_ingredients
        self.max_weight = max_weight
        self.weights = np.array(weights)
        self.tastes = np.array(tastes)
        self.incompatibilities = incompatibilities
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = inertia
        self.c1 = cognitive_weight
        self.c2 = social_weight

        # Inicialização das partículas (soluções binárias)
        self.particles = np.random.randint(0, 2, (n_particles, n_ingredients))
        self.velocities = np.random.uniform(-1, 1, (n_particles, n_ingredients))
        
        # Melhores posições e fitness
        self.pbest = self.particles.copy()
        self.pbest_fitness = np.array([self._fitness(p) for p in self.particles])
        self.gbest = self.pbest[np.argmax(self.pbest_fitness)]
        self.gbest_fitness = np.max(self.pbest_fitness)

    def _fitness(self, solution: np.ndarray) -> float:
        """Calcula o fitness (sabor total) da solução, penalizando inviabilidades."""
        total_weight = np.sum(self.weights * solution)
        total_taste = np.sum(self.tastes * solution)

        # Penaliza soluções que excedem o peso máximo
        if total_weight > self.max_weight:
            return -np.inf

        # Penaliza soluções com ingredientes incompatíveis
        for j, k in self.incompatibilities:
            if solution[j] == 1 and solution[k] == 1:
                return -np.inf

        return total_taste

    def _repair(self, solution: np.ndarray) -> np.ndarray:
        """Corrige soluções inviáveis removendo ingredientes conflitantes ou pesados."""
        # Remove conflitos de incompatibilidade (prioriza ingredientes com maior sabor)
        for j, k in self.incompatibilities:
            if solution[j] == 1 and solution[k] == 1:
                if self.tastes[j] > self.tastes[k]:
                    solution[k] = 0
                else:
                    solution[j] = 0

        # Remove ingredientes aleatórios se o peso total for excedido
        while np.sum(self.weights * solution) > self.max_weight:
            selected = np.where(solution == 1)[0]
            if len(selected) == 0:
                break
            # Remove o ingrediente com menor razão sabor/peso para otimização
            ratios = self.tastes[selected] / self.weights[selected]
            idx_to_remove = selected[np.argmin(ratios)]
            solution[idx_to_remove] = 0

        return solution

    def optimize(self) -> Tuple[np.ndarray, float]:
        """Executa o PSO e retorna a melhor solução e seu fitness."""
        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                # Atualiza velocidade
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (self.pbest[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social

                # Atualiza posição (convertendo velocidades para probabilidades com sigmoid)
                sigmoid = 1 / (1 + np.exp(-self.velocities[i]))
                self.particles[i] = np.where(sigmoid > random.random(), 1, 0)

                # Repara a solução
                self.particles[i] = self._repair(self.particles[i])

                # Atualiza pbest e gbest
                current_fitness = self._fitness(self.particles[i])
                if current_fitness > self.pbest_fitness[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_fitness[i] = current_fitness
                    if current_fitness > self.gbest_fitness:
                        self.gbest = self.particles[i].copy()
                        self.gbest_fitness = current_fitness

        return self.gbest, self.gbest_fitness


def load_instance(filename: str) -> Tuple[int, int, List[float], List[float], List[Tuple[int, int]]]:
    """Carrega uma instância do arquivo .dat com o formato fornecido."""
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    header = list(map(int, lines[0].split()))
    n_ingredients, max_weight = header[0], header[1]

    tastes = []
    weights = []
    incompat_start = 0

    for i, line in enumerate(lines[1:]):
        if line.startswith('1 2'):
            incompat_start = i + 1
            break
        tastes.extend(map(float, line.split()))

    if incompat_start == 0:
        tastes = list(map(float, ' '.join(lines[1:]).split()))

    if len(tastes) < n_ingredients:
        tastes += [0.0] * (n_ingredients - len(tastes))
    else:
        tastes = tastes[:n_ingredients]

    weights = [1.0] * n_ingredients

    incompatibilities = []
    for line in lines[incompat_start:]:
        parts = line.split()
        if len(parts) >= 2:
            j, k = map(int, parts[:2])
            incompatibilities.append((j - 1, k - 1))

    return n_ingredients, max_weight, weights, tastes, incompatibilities


def run_pso_on_instance(instance_file: str, runs: int = 30) -> dict:
    """Executa o PSO múltiplas vezes em uma instância e retorna estatísticas."""
    n_ingredients, max_weight, weights, tastes, incompatibilities = load_instance(instance_file)
    
    results = {
        'fitness_values': [],
        'best_solution': None,
        'best_fitness': -np.inf
    }
    
    for _ in range(runs):
        pso = PSO(n_ingredients, max_weight, weights, tastes, incompatibilities)
        solution, fitness = pso.optimize()
        
        results['fitness_values'].append(fitness)
        if fitness > results['best_fitness']:
            results['best_solution'] = solution
            results['best_fitness'] = fitness
    
    results['mean'] = np.mean(results['fitness_values'])
    results['min'] = np.min(results['fitness_values'])
    results['max'] = np.max(results['fitness_values'])
    results['std'] = np.std(results['fitness_values'])
    
    return results


def main():
    instances = [f"instances/ep{i:02d}.dat" for i in range(1, 11)]
    bkv = {
        'ep01': 2118, 'ep02': 1378, 'ep03': 2850, 'ep04': 2730, 'ep05': 2624,
        'ep06': 4690, 'ep07': 4440, 'ep08': 5020, 'ep09': 4568, 'ep10': 4390
    }
    
    all_results = {}
    total_instances = len(instances)
    
    for idx, instance in enumerate(instances, start=1):
        progress = (idx / total_instances) * 100
        print(f"[{progress:.1f}%] Executando PSO na instância {instance}...")
        
        start_time = time.time()
        results = run_pso_on_instance(instance)
        elapsed_time = time.time() - start_time
        
        all_results[instance] = results
        print(f"Resultados para {instance}:")
        print(f"  - Melhor fitness: {results['best_fitness']} (BKV: {bkv[instance.split('/')[-1].split('.')[0]]})")
        print(f"  - Média: {results['mean']:.2f} | Min: {results['min']} | Max: {results['max']} | Std: {results['std']:.2f}")
        print(f"  - Tempo de execução: {elapsed_time:.2f} segundos\n")
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([all_results[inst]['fitness_values'] for inst in instances], labels=[f"ep{i:02d}" for i in range(1, 11)])
    plt.title("Distribuição do Fitness por Instância (30 execuções)")
    plt.xlabel("Instâncias")
    plt.ylabel("Sabor Total")
    plt.grid(True)
    plt.savefig("boxplot_results.png")
    plt.show()
    
    with open("best_solutions.txt", 'w') as f:
        for inst in instances:
            f.write(f"{inst.split('/')[-1]}: {all_results[inst]['best_solution'].nonzero()[0].tolist()} (Fitness: {all_results[inst]['best_fitness']})\n")

if __name__ == "__main__":
    main()

import random
import copy
from time import time
from math import cos, pi, sqrt

class OtimizadorILS:
    def __init__(self, funcao_objetivo, limites, iteracoes_ils=100, iteracoes_busca_local=50, perturbacao=1.0):
        self.funcao_objetivo = funcao_objetivo
        self.limites = limites
        self.iteracoes_ils = iteracoes_ils
        self.iteracoes_busca_local = iteracoes_busca_local
        self.perturbacao = perturbacao

    def otimizar(self):
        solucao_atual = self.gerar_solucao_inicial()
        melhor_solucao = copy.deepcopy(solucao_atual)
        melhor_valor = self.funcao_objetivo(melhor_solucao)

        for _ in range(self.iteracoes_ils):
            solucao_busca_local = self.busca_local(solucao_atual)
            valor_busca_local = self.funcao_objetivo(solucao_busca_local)

            if valor_busca_local < melhor_valor:
                melhor_solucao = copy.deepcopy(solucao_busca_local)
                melhor_valor = valor_busca_local

            solucao_atual = self.perturbacao_solucao(melhor_solucao)

        return melhor_solucao

    def gerar_solucao_inicial(self):
        return [random.uniform(limite_minimo, limite_maximo) for limite_minimo, limite_maximo in self.limites]

    def busca_local(self, solucao):
        melhor_solucao = copy.deepcopy(solucao)
        melhor_valor = self.funcao_objetivo(melhor_solucao)

        for _ in range(self.iteracoes_busca_local):
            vizinho = self.gerar_vizinho(melhor_solucao)
            valor_vizinho = self.funcao_objetivo(vizinho)

            if valor_vizinho < melhor_valor:
                melhor_solucao = copy.deepcopy(vizinho)
                melhor_valor = valor_vizinho

        return melhor_solucao

    def gerar_vizinho(self, solucao):
        vizinho = copy.deepcopy(solucao)
        indice = random.randint(0, len(vizinho) - 1)

        vizinho[indice] += random.uniform(-self.perturbacao, self.perturbacao)

        # Garante que a nova solução esteja dentro dos limites
        vizinho[indice] = max(self.limites[indice][0], min(vizinho[indice], self.limites[indice][1]))
        return vizinho

    def perturbacao_solucao(self, solucao):
        solucao_perturbada = copy.deepcopy(solucao)
        for i in range(len(solucao_perturbada)):
            solucao_perturbada[i] += random.uniform(-self.perturbacao, self.perturbacao)
            solucao_perturbada[i] = max(self.limites[i][0], min(solucao_perturbada[i], self.limites[i][1]))
        return solucao_perturbada

# Função Objetivo
def funcao_objetivo(solucao):
    soma_quadrados = sum(x**2 for x in solucao)
    return 1 - cos(2 * pi * sqrt(soma_quadrados)) + 0.1 * sqrt(soma_quadrados)

# Problema A
limites_a = [[-100, 100], [-100, 100]]
otimizador_ils_a = OtimizadorILS(funcao_objetivo, limites_a)
solucao_a = otimizador_ils_a.otimizar()
print("Solução para o problema a):", solucao_a)
print("Valor da função objetivo:", funcao_objetivo(solucao_a))

# Problema B
limites_b = [[-20, 20], [-20, 20]]
otimizador_ils_b = OtimizadorILS(funcao_objetivo, limites_b)
solucao_b = otimizador_ils_b.otimizar()
print("Solução para o problema b):", solucao_b)
print("Valor da função objetivo:", funcao_objetivo(solucao_b))

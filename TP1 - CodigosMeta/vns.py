from time import time
from random import uniform, random
from math import cos, pi, sqrt

class OtimizadorVNS:
    def __init__(self, funcao_objetivo, limites, vizinhancas, probabilidade_ajuste = 0.5, limite_tempo = 2, limite_iteracoes = 100):
        self.funcao_objetivo = funcao_objetivo
        self.limites = limites
        self.vizinhancas = vizinhancas
        self.probabilidade_ajuste = probabilidade_ajuste
        self.limite_tempo = limite_tempo
        self.limite_iteracoes = limite_iteracoes

    def otimizar(self):
        tempo_inicial = time()
        solucao = self.obter_solucao_inicial()
        melhores_valores = [self.funcao_objetivo(solucao)]
        while not self.condicao_parada(tempo_inicial, melhores_valores):
            indice_vizinhanca = 0
            while not indice_vizinhanca == len(self.vizinhancas):
                proxima_solucao = self.ajustar(solucao, indice_vizinhanca)
                solucao, indice_vizinhanca = self.troca_vizinhanca(solucao, proxima_solucao, indice_vizinhanca)
                melhores_valores.append(self.funcao_objetivo(solucao))
        return solucao

    def troca_vizinhanca(self, solucao, proxima_solucao, indice_vizinhanca):
        if self.funcao_objetivo(proxima_solucao) < self.funcao_objetivo(solucao):
            return proxima_solucao, 1
        else:
            return solucao, indice_vizinhanca + 1

    def obter_solucao_inicial(self):
        return [uniform(limite_minimo, limite_maximo) for limite_minimo, limite_maximo in self.limites]

    def condicao_parada(self, tempo_inicial, melhores_valores):
        if time() - tempo_inicial > self.limite_tempo:
            return True
        if len(melhores_valores) <= self.limite_iteracoes:
            return False
        contador = 0
        for i in range(self.limite_iteracoes):
            if len(melhores_valores) > i + 1:
                valor_atual = melhores_valores[-1-i]
                valor_anterior = melhores_valores[-2-i]
                if valor_anterior / valor_atual <= 1.001:
                    contador += 1
        if contador >= self.limite_iteracoes:
            return True
        return False
    
    def ajustar(self, solucao, indice_vizinhanca):
        solucao_ajustada = solucao.copy()
        for indice, valor in enumerate(solucao_ajustada):
            if random() <= self.probabilidade_ajuste:
                while True:
                    solucao_ajustada[indice] = valor + uniform(-1, 1) * self.vizinhancas[indice_vizinhanca]
                    if self.limites[indice][0] <= solucao_ajustada[indice] <= self.limites[indice][1]:
                        break
        return solucao_ajustada

# Função Objetivo
def funcao_objetivo(solucao):
    dimensao = 2
    soma_quadrados = sum(x**2 for x in solucao)
    return 1 - cos(2 * pi * sqrt(soma_quadrados)) + 0.1 * sqrt(soma_quadrados)

#A
limites_a = [[-100, 100], [-100, 100]]
vizinhancas_a = [1, 10, 100]
otimizador_vns_a = OtimizadorVNS(funcao_objetivo, limites_a, vizinhancas_a)
solucao_a = otimizador_vns_a.otimizar()
print("Solução para o problema a):", solucao_a)
print("Valor da função objetivo:", funcao_objetivo(solucao_a))

# B
limites_b = [[-20, 20], [-20, 20]]
vizinhancas_b = [0.1, 1, 10]
otimizador_vns_b = OtimizadorVNS(funcao_objetivo, limites_b, vizinhancas_b)
solucao_b = otimizador_vns_b.otimizar()
print("Solução para o problema b):", solucao_b)
print("Valor da função objetivo:", funcao_objetivo(solucao_b))
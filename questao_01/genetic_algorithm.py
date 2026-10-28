"""Etapa B — Algoritmo Genético para o catálogo completo.

Cromossomo: vetor de n inteiros; gene i = índice da opção de preço do SKU i
(0..L-1 = nível de preço, L = suspender venda). Só genes permitidos (margem
mínima) são sorteados, então todo indivíduo respeita o piso de margem; o limite
de SKUs com desconto é tratado por penalidade na função de fitness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .modelo import TabelaQ1, avaliar_populacao


@dataclass
class ConfigGA:
    populacao: int = 120
    geracoes_max: int = 300
    taxa_crossover: float = 0.9
    taxa_mutacao: float = 0.05     # probabilidade por gene
    elitismo: int = 4              # melhores copiados intactos
    torneio: int = 3               # tamanho do torneio de seleção
    paciencia: int = 60            # parada: gerações sem melhora
    seed: int = 0


@dataclass
class ResultadoGA:
    melhor: np.ndarray
    melhor_fitness: float
    historico_melhor: list[float] = field(default_factory=list)
    historico_media: list[float] = field(default_factory=list)
    geracoes: int = 0


def _sortear_genes(rng, opcoes: list[np.ndarray], tamanho: int) -> np.ndarray:
    """Gera 'tamanho' indivíduos aleatórios, cada gene dentro das opções permitidas."""
    n = len(opcoes)
    pop = np.empty((tamanho, n), dtype=np.int64)
    for i, op in enumerate(opcoes):
        pop[:, i] = rng.choice(op, size=tamanho)
    return pop


def _torneio(rng, fitness: np.ndarray, k: int, quantos: int) -> np.ndarray:
    competidores = rng.integers(0, len(fitness), size=(quantos, k))
    vencedor = np.argmax(fitness[competidores], axis=1)
    return competidores[np.arange(quantos), vencedor]


def algoritmo_genetico(tab: TabelaQ1, cfg: ConfigGA, semente_inicial: np.ndarray | None = None) -> ResultadoGA:
    rng = np.random.default_rng(cfg.seed)
    opcoes = [np.flatnonzero(tab.permitido[i]) for i in range(tab.n)]

    pop = _sortear_genes(rng, opcoes, cfg.populacao)
    # indivíduo "preço atual" (nível 1.00 quando viável) ajuda a convergência
    if semente_inicial is not None:
        pop[0] = semente_inicial
    fit = avaliar_populacao(tab, pop)

    res = ResultadoGA(melhor=pop[np.argmax(fit)].copy(), melhor_fitness=float(fit.max()))
    sem_melhora = 0
    n_filhos = cfg.populacao - cfg.elitismo

    for g in range(cfg.geracoes_max):
        # elitismo
        elite = pop[np.argsort(fit)[-cfg.elitismo:]]

        # seleção por torneio
        pais = pop[_torneio(rng, fit, cfg.torneio, 2 * n_filhos)]
        pai_a, pai_b = pais[:n_filhos], pais[n_filhos:]

        # crossover uniforme
        faz_cross = rng.random(n_filhos) < cfg.taxa_crossover
        mascara = (rng.random((n_filhos, tab.n)) < 0.5) & faz_cross[:, None]
        filhos = np.where(mascara, pai_b, pai_a)

        # mutação: re-sorteia o gene dentro das opções permitidas daquele SKU
        muta = rng.random((n_filhos, tab.n)) < cfg.taxa_mutacao
        if muta.any():
            novos = _sortear_genes(rng, opcoes, n_filhos)
            filhos = np.where(muta, novos, filhos)

        pop = np.vstack([elite, filhos])
        fit = avaliar_populacao(tab, pop)               # domina: O(P * n)

        j = int(np.argmax(fit))
        res.historico_melhor.append(float(fit[j]))
        res.historico_media.append(float(np.mean(fit[np.isfinite(fit)])))
        if fit[j] > res.melhor_fitness + 1e-9:
            res.melhor, res.melhor_fitness = pop[j].copy(), float(fit[j])
            sem_melhora = 0
        else:
            sem_melhora += 1
        res.geracoes = g + 1
        if sem_melhora >= cfg.paciencia:
            break
    return res

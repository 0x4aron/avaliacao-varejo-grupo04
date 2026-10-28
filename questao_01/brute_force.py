"""Etapa A — Brute Force: ótimo exato num subconjunto pequeno (benchmark do GA).

Enumera TODAS as combinações de opções permitidas de cada SKU do subconjunto.
Número de combinações = produto de |opções permitidas de cada SKU| <= (L+1)^k.
"""
from __future__ import annotations

import itertools

import numpy as np

from .modelo import TabelaQ1, avaliar_populacao

TAMANHO_LOTE = 50_000


def opcoes_permitidas(tab: TabelaQ1) -> list[np.ndarray]:
    return [np.flatnonzero(tab.permitido[i]) for i in range(tab.n)]


def numero_combinacoes(tab: TabelaQ1) -> int:
    return int(np.prod([len(o) for o in opcoes_permitidas(tab)], dtype=np.int64))


def brute_force(tab: TabelaQ1) -> tuple[np.ndarray, float, int]:
    """Retorna (melhor_escolha, melhor_fitness, combinacoes_avaliadas).

    As combinações são geradas por itertools.product e avaliadas em lotes
    vetorizados (avaliar_populacao), para não materializar tudo na memória.
    """
    produto = itertools.product(*opcoes_permitidas(tab))
    melhor_valor, melhor = -np.inf, None
    avaliadas = 0
    while True:
        lote = list(itertools.islice(produto, TAMANHO_LOTE))
        if not lote:
            break
        pop = np.array(lote, dtype=np.int64)
        valores = avaliar_populacao(tab, pop)          # domina: O(lote * k)
        j = int(np.argmax(valores))
        if valores[j] > melhor_valor:
            melhor_valor, melhor = float(valores[j]), pop[j].copy()
        avaliadas += len(pop)
    return melhor, melhor_valor, avaliadas


def selecionar_subconjunto(tab: TabelaQ1, k: int, seed: int) -> list[int]:
    """Escolhe k SKUs para o benchmark entre os que têm >= 3 opções viáveis
    (subproblema não trivial); sorteio reprodutível pela seed."""
    rng = np.random.default_rng(seed)
    candidatos = [i for i in range(tab.n) if tab.permitido[i].sum() >= 3]
    if len(candidatos) < k:
        candidatos = list(range(tab.n))
    return sorted(rng.choice(candidatos, size=k, replace=False).tolist())

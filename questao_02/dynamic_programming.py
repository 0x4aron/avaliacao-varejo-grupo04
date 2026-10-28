"""Etapa B — Programação Dinâmica (Bottom-Up) por SKU.

Formulação (um SKU):
    Estado    (t, s): semana t = 0..T e estoque remanescente s = 0..S
    Decisão   j: nível de preço (ou suspender venda, j = L)
    Transição s' = s - min(s, d_k), onde d_k é o k-ésimo quantil de demanda
              ESTIMADO PELO MONTE CARLO para o nível j (K buckets equiprováveis)
    Recompensa r = peso_t * min(s, d_k) * margem_unit_media[j]   (margem também do MC)
    Objetivo  V(t, s) = max_j  (1/K) * sum_k [ r + V(t+1, s') ]
              V(T, s) = valor_residual * custo_unitario * s
    peso_t = (1 + urgencia) ** (-t): com urgencia > 0 o caixa que entra antes vale
    mais, o que antecipa receita para as dívidas das semanas 4 e 8.

Reutilização de subproblemas: a linha V[t+1, :] é calculada uma vez e consultada
por TODOS os pares (s, j, k) da semana t — são T*(S+1) estados resolvidos e
T*(S+1)*L*K consultas. O contador 'consultas' expõe isso.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResultadoDP:
    V: np.ndarray              # (T+1, S+1)
    politica: np.ndarray       # (T, S+1) índice do nível ótimo
    estados: int
    consultas: int


def dp_sku(estoque_inicial: int, T: int, quantis: np.ndarray, margem: np.ndarray,
           urgencia: float = 0.0, valor_residual_unit: float = 0.0,
           permitido: np.ndarray | None = None) -> ResultadoDP:
    """quantis: (L, K) inteiros; margem: (L,) contribuição unitária média."""
    L, K = quantis.shape
    S = int(estoque_inicial)
    s = np.arange(S + 1)
    V = np.zeros((T + 1, S + 1))
    V[T] = valor_residual_unit * s
    politica = np.zeros((T, S + 1), dtype=np.int64)
    if permitido is None:
        permitido = np.ones(L, dtype=bool)
    consultas = 0

    for t in range(T - 1, -1, -1):                       # bottom-up: da última semana para a primeira
        peso = (1 + urgencia) ** (-t)
        # opção "suspender": não vende, estoque fica igual
        melhor = V[t + 1].copy()
        melhor_j = np.full(S + 1, L, dtype=np.int64)
        for j in range(L):
            if not permitido[j]:
                continue
            vendas = np.minimum(s[:, None], quantis[j][None, :])          # (S+1, K)
            valor = peso * vendas * margem[j] + V[t + 1][s[:, None] - vendas]  # reutiliza V[t+1]
            esperado = valor.mean(axis=1)                                   # domina: O(S*K) por j
            consultas += vendas.size
            troca = esperado > melhor + 1e-9
            melhor = np.where(troca, esperado, melhor)
            melhor_j = np.where(troca, j, melhor_j)
        V[t] = melhor
        politica[t] = melhor_j
    return ResultadoDP(V=V, politica=politica, estados=T * (S + 1), consultas=consultas)


def trajetoria_esperada(res: ResultadoDP, estoque_inicial: int, demanda_media: np.ndarray,
                        uplift_semana: np.ndarray | None = None) -> tuple[list[int], list[float]]:
    """Reconstrói o caminho de decisões a partir de (0, estoque_inicial) seguindo a
    política ótima com a demanda MÉDIA do MC. Retorna (níveis, estoque no início de cada semana)."""
    T = res.politica.shape[0]
    L = len(demanda_media)
    s = float(estoque_inicial)
    niveis, estoques = [], []
    for t in range(T):
        j = int(res.politica[t, int(round(s))])
        niveis.append(j)
        estoques.append(s)
        if j < L:
            d = demanda_media[j] * (1 + (uplift_semana[t] if uplift_semana is not None else 0.0))
            s = max(0.0, s - d)
    return niveis, estoques

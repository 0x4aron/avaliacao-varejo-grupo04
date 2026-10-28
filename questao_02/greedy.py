"""Etapa C — Greedy semanal de destaques (mídia / frete promocional).

Entrada: a trajetória da DP (nível de preço e estoque esperado por semana).
A cada semana há no máximo 'max_destaques' vagas e o orçamento promocional
semanal. Um destaque eleva a demanda do SKU em 'uplift' e custa
'frac_custo' x receita esperada do SKU naquela semana (no preço da DP).

Regra gulosa: ordenar os SKUs pelo ganho esperado de contribuição por real
gasto e preencher vagas/orçamento nessa ordem, pulando quem não cabe.
Custo: O(T * n log n).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfigDestaque:
    max_destaques: int = 4
    uplift: float = 0.20
    frac_custo: float = 0.04


def alocar_destaques(niveis: np.ndarray, estoques: np.ndarray, preco: np.ndarray, demanda_media: np.ndarray,
                     margem: np.ndarray, orcamento_semanal: float, cfg: ConfigDestaque):
    """niveis, estoques: (n, T) da trajetória da DP; preco/demanda_media/margem: (n, L).

    Retorna (destaques bool (T, n), custo (T, n), tabela de decisões)."""
    n, T = niveis.shape
    L = preco.shape[1]
    destaques = np.zeros((T, n), dtype=bool)
    custo = np.zeros((T, n))
    decisoes = []
    for t in range(T):
        candidatos = []
        for i in range(n):
            j = niveis[i, t]
            if j >= L or estoques[i, t] <= 0:
                continue
            base = min(estoques[i, t], demanda_media[i, j])
            com = min(estoques[i, t], demanda_media[i, j] * (1 + cfg.uplift))
            ganho = (com - base) * margem[i, j]
            c = cfg.frac_custo * base * preco[i, j]
            if ganho > 0 and c > 0:
                candidatos.append((ganho / c, i, ganho, c))
        candidatos.sort(reverse=True)                      # domina: O(n log n) por semana
        gasto, usados = 0.0, 0
        for razao, i, ganho, c in candidatos:
            if usados >= cfg.max_destaques:
                break
            if razao <= 1.0 or gasto + c > orcamento_semanal:
                continue
            destaques[t, i] = True
            custo[t, i] = c
            gasto += c
            usados += 1
            decisoes.append({"semana": t + 1, "sku_idx": i, "ganho_esperado": round(ganho, 2),
                             "custo": round(c, 2), "razao": round(razao, 3)})
    return destaques, custo, decisoes

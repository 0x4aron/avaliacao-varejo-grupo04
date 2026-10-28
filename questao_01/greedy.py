"""Etapa C — Greedy: distribui o orçamento de incentivo SOBRE a política do GA.

Incentivo = cupom/cashback em degraus de 2,5% do preço definido pelo GA,
pago pelo orçamento promocional. O cupom reduz o preço efetivo ao cliente
(eleva a demanda pela elasticidade), mas a loja continua recebendo o preço
do GA; o custo do cupom sai do orçamento.

Benefício de um degrau = ganho de caixa operacional + redução do custo de
estoque parado. Regra gulosa: sempre aplicar o próximo degrau com maior
benefício marginal por real investido, enquanto houver orçamento e a razão
for maior que 'razao_minima'. Usa heap: O(D log n), D = degraus aplicados.

O orçamento promocional é tratado como verba já provisionada (separada do
caixa da dívida). O main reporta também o caixa LÍQUIDO do incentivo para
mostrar quanto da verba retorna em caixa (ver docs/decisoes.md).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .modelo import TabelaQ1


@dataclass
class ConfigIncentivo:
    degrau: float = 0.025
    max_incentivo: float = 0.10
    razao_minima: float = 0.0   # >0: só investe se o degrau gera benefício positivo


def _efeito(r: dict, preco: float, x: float, tc: float) -> tuple[float, float, float]:
    """(unidades, benefício, custo) do SKU com preço do GA 'preco' e cupom x."""
    if preco <= 0:
        return 0.0, 0.0, 0.0
    efetivo = preco * (1 - x)
    dem = r["demanda_base_90d"] * (efetivo / r["preco_atual"]) ** (-r["elasticidade"])
    u = min(r["estoque"], dem)
    margem = preco * (1 - r["taxa_marketplace"]) - r["custo_unitario"] - r["custo_logistico"]
    parado = (r["estoque"] - u) * r["custo_unitario"] * tc * r["idade_estoque_dias"] / 360
    return u, u * margem - parado, u * preco * x


def alocar_incentivo(tab: TabelaQ1, escolha: np.ndarray, catalogo: pd.DataFrame,
                     orcamento: float, cfg: ConfigIncentivo = ConfigIncentivo()) -> pd.DataFrame:
    linhas = catalogo.to_dict("records")
    precos = tab.preco[np.arange(tab.n), escolha]
    tc = tab.params.taxa_carregamento
    nivel = np.zeros(tab.n)                      # incentivo atual de cada SKU
    estado = [_efeito(linhas[i], precos[i], 0.0, tc) for i in range(tab.n)]

    def proximo(i: int):
        x = nivel[i] + cfg.degrau
        if x > cfg.max_incentivo + 1e-9 or precos[i] <= 0:
            return None
        novo = _efeito(linhas[i], precos[i], x, tc)
        d_benef = novo[1] - estado[i][1]
        d_custo = novo[2] - estado[i][2]
        if d_custo <= 0:
            return None
        return d_benef / d_custo, d_benef, d_custo, x, novo

    heap = []
    for i in range(tab.n):
        p = proximo(i)
        if p:
            heapq.heappush(heap, (-p[0], i, p))

    gasto = 0.0
    while heap:
        _, i, (razao, d_benef, d_custo, x, novo) = heapq.heappop(heap)
        if razao <= cfg.razao_minima:
            break                                   # os demais são ainda piores
        if gasto + d_custo > orcamento:
            continue                                # não cabe; tenta o próximo
        gasto += d_custo
        nivel[i], estado[i] = x, novo
        p = proximo(i)
        if p:
            heapq.heappush(heap, (-p[0], i, p))

    base = [_efeito(linhas[i], precos[i], 0.0, tc) for i in range(tab.n)]
    return pd.DataFrame({
        "sku": tab.skus,
        "preco_ga": precos.round(2),
        "incentivo_pct": (nivel * 100).round(1),
        "unidades_antes": [round(b[0], 1) for b in base],
        "unidades_depois": [round(e[0], 1) for e in estado],
        "beneficio_antes": [round(b[1], 2) for b in base],
        "beneficio_depois": [round(e[1], 2) for e in estado],
        "custo_incentivo": [round(e[2], 2) for e in estado],
    })

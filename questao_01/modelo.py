"""Modelo econômico da Questão 1 (Operação Caixa 90).

Demanda com elasticidade constante:
    demanda(p) = demanda_base_90d * (p / preco_atual) ** (-elasticidade)
Unidades vendidas:  u = min(estoque, demanda(p))
Caixa do SKU:       u * (p * (1 - taxa_marketplace) - custo_unitario - custo_logistico)

Com p = preco_atual e margens negativas truncadas em zero, a soma reproduz
exatamente o 'caixa_base_estimado' usado pelo notebook gerador para definir a
meta e a dívida (ver tests/test_q1.py::test_caixa_base_reproduz_gerador).

Estrutura central: a TABELA de contribuição (n_skus x n_opcoes) é calculada uma
única vez; BF, GA e Greedy apenas consultam essa tabela.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INVIAVEL = -np.inf


@dataclass
class ParametrosQ1:
    margem_minima: float
    niveis: np.ndarray                 # multiplicadores de preço (ex.: 0.80 ... 1.10)
    max_frac_desconto: float = 0.30    # no máximo 30% dos SKUs com preço < atual
    taxa_carregamento: float = 0.25    # custo anual de capital + obsolescência (% do custo)
    fator_giro: float = 1.0            # giro mínimo = fator x giro da política atual
    penalidade_giro: float = 5.0       # por R$ de capital não liberado abaixo do mínimo
    penalidade_violacao: float = 1e6   # por SKU acima do limite de descontos

    @property
    def n_opcoes(self) -> int:
        # último índice = "suspender venda" (0 unidades, 0 caixa)
        return len(self.niveis) + 1

    @property
    def idx_suspender(self) -> int:
        return len(self.niveis)


def parametros_de_config(config: dict, **sobrescritas) -> ParametrosQ1:
    p = ParametrosQ1(
        margem_minima=float(config["margem_minima_percentual"]),
        niveis=np.array(config["niveis_preco_relativos"], dtype=float),
    )
    for k, v in sobrescritas.items():
        if v is not None:
            setattr(p, k, v)
    return p


@dataclass
class TabelaQ1:
    """Tabelas pré-computadas, uma linha por SKU e uma coluna por opção de preço."""
    skus: list[str]
    preco: np.ndarray        # (n, L+1)
    unidades: np.ndarray     # (n, L+1)
    caixa: np.ndarray        # (n, L+1)  caixa operacional
    estoque_parado: np.ndarray  # (n, L+1) custo de carregamento do que sobra
    fitness: np.ndarray      # (n, L+1)  caixa - custo de carregamento; -inf se fora da margem
    permitido: np.ndarray    # (n, L+1)  bool
    desconto: np.ndarray     # (L+1,)    bool: opção é desconto (< 1.0)
    capital: np.ndarray      # (n, L+1)  capital liberado = unidades * custo_unitario
    giro_minimo: float       # R$ de custo que a política precisa girar em 90 dias
    params: ParametrosQ1

    @property
    def n(self) -> int:
        return len(self.skus)


def construir_tabela(catalogo: pd.DataFrame, params: ParametrosQ1) -> TabelaQ1:
    n, L = len(catalogo), len(params.niveis)
    p0 = catalogo["preco_atual"].to_numpy()[:, None]
    preco = np.zeros((n, L + 1))
    preco[:, :L] = p0 * params.niveis[None, :]

    elast = catalogo["elasticidade"].to_numpy()[:, None]
    dem_base = catalogo["demanda_base_90d"].to_numpy()[:, None]
    estoque = catalogo["estoque"].to_numpy()[:, None].astype(float)
    taxa = catalogo["taxa_marketplace"].to_numpy()[:, None]
    custo = catalogo["custo_unitario"].to_numpy()[:, None]
    logist = catalogo["custo_logistico"].to_numpy()[:, None]
    idade = catalogo["idade_estoque_dias"].to_numpy()[:, None]

    unidades = np.zeros((n, L + 1))
    unidades[:, :L] = np.minimum(estoque, dem_base * (preco[:, :L] / p0) ** (-elast))

    margem_unit = np.zeros((n, L + 1))
    margem_unit[:, :L] = preco[:, :L] * (1 - taxa) - custo - logist
    caixa = unidades * margem_unit

    sobra = estoque - unidades
    estoque_parado = sobra * custo * params.taxa_carregamento * (idade / 360.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        margem_pct = np.where(preco > 0, margem_unit / preco, 0.0)
    permitido = margem_pct >= params.margem_minima
    permitido[:, L] = True  # suspender sempre é permitido

    fitness = np.where(permitido, caixa - estoque_parado, INVIAVEL)
    desconto = np.zeros(L + 1, dtype=bool)
    desconto[:L] = params.niveis < 1.0

    capital = unidades * custo
    giro_atual = sum(capital[i, _opcao_referencia(permitido[i], params)] for i in range(n))

    return TabelaQ1(
        skus=catalogo["sku"].tolist(), preco=preco, unidades=unidades, caixa=caixa,
        estoque_parado=estoque_parado, fitness=fitness, permitido=permitido,
        desconto=desconto, capital=capital,
        giro_minimo=float(params.fator_giro * giro_atual), params=params,
    )


def _opcao_referencia(permitido_sku: np.ndarray, params: ParametrosQ1) -> int:
    """Opção que representa 'manter a política atual': nível 1.00 se viável;
    senão o menor preço viável acima dele; senão suspender."""
    i_atual = int(np.argmin(np.abs(params.niveis - 1.0)))
    for j in range(i_atual, len(params.niveis)):
        if permitido_sku[j]:
            return j
    return params.idx_suspender


def limite_descontos(tab: TabelaQ1) -> int:
    return int(np.floor(tab.params.max_frac_desconto * tab.n))


def avaliar(tab: TabelaQ1, escolha: np.ndarray) -> float:
    """Fitness de UMA política (vetor de índices de opção). O(n)."""
    return float(avaliar_populacao(tab, escolha[None, :])[0])


def avaliar_populacao(tab: TabelaQ1, pop: np.ndarray) -> np.ndarray:
    """Fitness vetorizado de P políticas de uma vez (P x n). O(P*n)."""
    linhas = np.arange(tab.n)[None, :]
    valor = tab.fitness[linhas, pop].sum(axis=1)
    excesso = np.maximum(0, tab.desconto[pop].sum(axis=1) - limite_descontos(tab))
    falta_giro = np.maximum(0.0, tab.giro_minimo - tab.capital[linhas, pop].sum(axis=1))
    return (valor - tab.params.penalidade_violacao * excesso
            - tab.params.penalidade_giro * falta_giro)


def avaliar_populacao_loop(tab: TabelaQ1, pop: np.ndarray, catalogo: pd.DataFrame) -> np.ndarray:
    """Versão ANTES da otimização: recalcula demanda e caixa em laços Python
    a cada avaliação (sem a tabela pré-computada). Mesmo resultado, ~100x mais lenta.
    Mantida apenas para a comparação antes/depois em docs/desempenho.md.
    """
    p = tab.params
    lim = limite_descontos(tab)
    linhas = catalogo.to_dict("records")
    out = np.empty(len(pop))
    for k, ind in enumerate(pop):
        total, n_desc, giro = 0.0, 0, 0.0
        for i, op in enumerate(ind):
            r = linhas[i]
            if op == p.idx_suspender:
                u, m, preco = 0.0, 0.0, 0.0
            else:
                preco = r["preco_atual"] * p.niveis[op]
                m = preco * (1 - r["taxa_marketplace"]) - r["custo_unitario"] - r["custo_logistico"]
                if m / preco < p.margem_minima:
                    total = INVIAVEL
                    break
                dem = r["demanda_base_90d"] * (preco / r["preco_atual"]) ** (-r["elasticidade"])
                u = min(r["estoque"], dem)
                n_desc += int(p.niveis[op] < 1.0)
            giro += u * r["custo_unitario"]
            sobra = r["estoque"] - u
            total += u * m - sobra * r["custo_unitario"] * p.taxa_carregamento * r["idade_estoque_dias"] / 360
        out[k] = (total - p.penalidade_violacao * max(0, n_desc - lim)
                  - p.penalidade_giro * max(0.0, tab.giro_minimo - giro))
    return out


def resumo_politica(tab: TabelaQ1, escolha: np.ndarray, catalogo: pd.DataFrame) -> pd.DataFrame:
    linhas = np.arange(tab.n)
    rotulos = [f"{x:.2f}" for x in tab.params.niveis] + ["suspenso"]
    df = pd.DataFrame({
        "sku": tab.skus,
        "categoria": catalogo["categoria"].to_numpy(),
        "nivel": [rotulos[e] for e in escolha],
        "preco": tab.preco[linhas, escolha].round(2),
        "unidades": tab.unidades[linhas, escolha].round(1),
        "estoque": catalogo["estoque"].to_numpy(),
        "caixa": tab.caixa[linhas, escolha].round(2),
    })
    df["estoque_restante"] = (df["estoque"] - df["unidades"]).round(1)
    receita = tab.preco[linhas, escolha] * tab.unidades[linhas, escolha]
    df["receita"] = receita.round(2)
    return df

"""Etapa A — Monte Carlo (construção) e Etapa D — Monte Carlo (validação).

Modelo semanal de um SKU i ao preço p, num cenário aleatório:
    preco_conc = preco_concorrente_base * mult_conc * LogNormal(0, sigma_conc)
    demanda    = demanda_base_semanal * (p / preco_atual) ** (-elasticidade)
                 * (preco_conc / preco_concorrente_base) ** elasticidade_cruzada
                 * LogNormal(0, sigma_demanda)
    custo_log  = custo_logistico_base * LogNormal(0, sigma_log)
Os log-normais têm média 1 (mu = -sigma^2/2): sigma mede dispersão relativa,
como descrito no config ("parâmetros de dispersão relativos").

Três fontes de incerteza: demanda, preço do concorrente e custo logístico.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ParametrosQ2:
    niveis: np.ndarray
    sigma_demanda: float
    sigma_conc: float
    sigma_log: float
    elasticidade_cruzada: float = 0.8
    mult_concorrente: float = 1.0      # cenário: concorrente -8% => 0.92
    mult_custo_log: float = 1.0        # cenário: custo logístico +15% => 1.15
    n_quantis: int = 7                 # buckets de demanda passados à DP

    @property
    def idx_suspender(self) -> int:
        return len(self.niveis)


def parametros_de_config(config: dict, **sobrescritas) -> ParametrosQ2:
    v = config["variabilidade"]
    p = ParametrosQ2(
        niveis=np.array(config["niveis_preco_relativos"], dtype=float),
        sigma_demanda=float(v["sigma_demanda"]),
        sigma_conc=float(v["sigma_preco_concorrente"]),
        sigma_log=float(v["sigma_custo_logistico"]),
    )
    for k, val in sobrescritas.items():
        if val is not None:
            setattr(p, k, val)
    return p


def _lognormal(rng, sigma: float, size) -> np.ndarray:
    return rng.lognormal(mean=-sigma ** 2 / 2, sigma=sigma, size=size)


def sortear_choques(rng, n_cen: int, n_skus: int, n_semanas: int, p: ParametrosQ2) -> dict:
    """Choques multiplicativos independentes por cenário, semana e SKU."""
    forma = (n_cen, n_semanas, n_skus)
    return {
        "demanda": _lognormal(rng, p.sigma_demanda, forma),
        "concorrente": p.mult_concorrente * _lognormal(rng, p.sigma_conc, forma),
        "logistica": p.mult_custo_log * _lognormal(rng, p.sigma_log, forma),
    }


def demanda(cat_i: dict, preco, choque_dem, choque_conc, p: ParametrosQ2):
    return (cat_i["demanda_base_semanal"]
            * (preco / cat_i["preco_atual"]) ** (-cat_i["elasticidade"])
            * choque_conc ** p.elasticidade_cruzada
            * choque_dem)


def estatisticas_construcao(catalogo: pd.DataFrame, p: ParametrosQ2, n_cen: int, seed: int) -> dict:
    """MC de construção: para cada SKU e nível de preço estima, a partir de n_cen
    cenários, a distribuição da demanda semanal (quantis) e a contribuição
    unitária média. Esta tabela é a ENTRADA da Programação Dinâmica.

    Retorna arrays (n_skus, L): preco, demanda_media, margem_unit_media,
    e (n_skus, L, K) demanda_quantis (inteiros, K buckets equiprováveis).
    """
    rng = np.random.default_rng(seed)
    n, L, K = len(catalogo), len(p.niveis), p.n_quantis
    ch = sortear_choques(rng, n_cen, n, 1, p)
    linhas = catalogo.to_dict("records")

    preco = np.zeros((n, L))
    dem_media = np.zeros((n, L))
    dem_quantis = np.zeros((n, L, K), dtype=np.int64)
    margem_media = np.zeros((n, L))
    probs = (np.arange(K) + 0.5) / K                       # ponto médio de cada bucket
    for i, r in enumerate(linhas):
        log_medio = r["custo_logistico_base"] * ch["logistica"][:, 0, i].mean()
        for j, nivel in enumerate(p.niveis):
            pr = r["preco_atual"] * nivel
            d = demanda(r, pr, ch["demanda"][:, 0, i], ch["concorrente"][:, 0, i], p)
            preco[i, j] = pr
            dem_media[i, j] = d.mean()
            dem_quantis[i, j] = np.round(np.quantile(d, probs)).astype(np.int64)
            margem_media[i, j] = pr * (1 - r["taxa_marketplace"]) - r["custo_unitario"] - log_medio
    return {"preco": preco, "demanda_media": dem_media, "demanda_quantis": dem_quantis,
            "margem_unit_media": margem_media, "n_cenarios": n_cen}


@dataclass
class ResultadoValidacao:
    caixa_final: np.ndarray            # (n_cen,)
    pagou: np.ndarray                  # (n_cen, n_dividas) bool
    caixa_semanal: np.ndarray          # (n_cen, T)
    estoque_final: np.ndarray          # (n_cen, n_skus)
    receita: np.ndarray                # (n_cen,)
    contribuicao: np.ndarray           # (n_cen,)


def validar_plano(catalogo: pd.DataFrame, config: dict, p: ParametrosQ2, politicas: list[np.ndarray],
                  destaques: np.ndarray, custo_destaque: np.ndarray, uplift: float,
                  n_cen: int, seed: int) -> ResultadoValidacao:
    """MC de validação: simula o plano COMPLETO (política da DP + destaques do Greedy)
    em cenários novos (seed diferente da construção). Vetorizado nos cenários.

    politicas[i][t, s] = índice do nível de preço do SKU i na semana t com estoque s
    destaques[t, i]    = True se o SKU i recebe destaque na semana t
    """
    rng = np.random.default_rng(seed)
    n, T = len(catalogo), int(config["numero_semanas"])
    ch = sortear_choques(rng, n_cen, n, T, p)
    linhas = catalogo.to_dict("records")
    vencimentos = {int(k.split("_")[1]): v for k, v in config["dividas"].items()}
    venc_ordenados = sorted(vencimentos)

    estoque = np.tile(catalogo["estoque_inicial"].to_numpy().astype(np.int64), (n_cen, 1))
    caixa = np.full(n_cen, float(config["caixa_inicial"]))
    pagou = np.zeros((n_cen, len(venc_ordenados)), dtype=bool)
    caixa_sem = np.zeros((n_cen, T))
    receita = np.zeros(n_cen)
    contrib = np.zeros(n_cen)
    niveis_ext = np.append(p.niveis, 0.0)                 # índice L = suspenso

    for t in range(T):
        for i, r in enumerate(linhas):
            s = estoque[:, i]
            j = politicas[i][t, s]                           # decisão adaptativa (estado real)
            suspenso = j == p.idx_suspender
            preco = r["preco_atual"] * niveis_ext[j]
            d = demanda(r, np.where(suspenso, r["preco_atual"], preco),
                        ch["demanda"][:, t, i], ch["concorrente"][:, t, i], p)
            if destaques[t, i]:
                d = d * (1 + uplift)
            vendas = np.where(suspenso, 0, np.minimum(s, np.round(d).astype(np.int64)))
            logist = r["custo_logistico_base"] * ch["logistica"][:, t, i]
            m = preco * (1 - r["taxa_marketplace"]) - r["custo_unitario"] - logist
            caixa += vendas * m
            receita += vendas * preco
            contrib += vendas * m
            estoque[:, i] = s - vendas
        caixa -= destaques[t].astype(float) @ custo_destaque[t]
        semana = t + 1
        if semana in vencimentos:
            k = venc_ordenados.index(semana)
            ok = caixa >= vencimentos[semana]
            pagou[:, k] = ok
            caixa = np.where(ok, caixa - vencimentos[semana], caixa)
        caixa_sem[:, t] = caixa
    return ResultadoValidacao(caixa_final=caixa, pagou=pagou, caixa_semanal=caixa_sem,
                              estoque_final=estoque, receita=receita, contribuicao=contrib)

"""Questão 1 — Operação Caixa 90.  Fluxo: Brute Force -> GA -> Greedy.

Exemplos:
    python questao_01/main.py --modo completo
    python questao_01/main.py --meta-mult 1.10            # cenário-surpresa
    python questao_01/main.py --custo-log-mult 1.15 --orcamento-mult 0.75
    python questao_01/main.py --remover-sku Q1-005 --ga-pop 200 --ga-mut 0.10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comum.dados import carregar_q1, remover_skus, salvar_json  # noqa: E402
from comum.medicao import medir  # noqa: E402
from questao_01.brute_force import brute_force, numero_combinacoes, selecionar_subconjunto  # noqa: E402
from questao_01.genetic_algorithm import ConfigGA, algoritmo_genetico  # noqa: E402
from questao_01.greedy import alocar_incentivo  # noqa: E402
from questao_01.modelo import (avaliar_populacao, avaliar_populacao_loop, construir_tabela,  # noqa: E402
                               parametros_de_config, resumo_politica)

PASTA_RES = Path(__file__).resolve().parent / "resultados"


def argumentos(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None, help="padrão: seed_final do manifesto")
    ap.add_argument("--modo", choices=["rapido", "completo"], default="rapido",
                    help="completo inclui escalabilidade e otimização antes/depois")
    ap.add_argument("--subconjunto", type=int, default=7, help="nº de SKUs do Brute Force (>= 6)")
    ap.add_argument("--meta-mult", type=float, default=1.0)
    ap.add_argument("--custo-log-mult", type=float, default=1.0)
    ap.add_argument("--orcamento-mult", type=float, default=1.0)
    ap.add_argument("--remover-sku", nargs="*", default=None)
    ap.add_argument("--max-frac-desconto", type=float, default=None)
    ap.add_argument("--fator-giro", type=float, default=None)
    ap.add_argument("--ga-pop", type=int, default=120)
    ap.add_argument("--ga-mut", type=float, default=0.05)
    ap.add_argument("--ga-geracoes", type=int, default=300)
    ap.add_argument("--saida", type=Path, default=PASTA_RES)
    return ap.parse_args(argv)


def aplicar_cenario(catalogo: pd.DataFrame, config: dict, a) -> tuple[pd.DataFrame, dict]:
    """Cenário-surpresa aplicado em memória (os arquivos de data/ não mudam)."""
    catalogo = remover_skus(catalogo, a.remover_sku).copy()
    catalogo["custo_logistico"] = catalogo["custo_logistico"] * a.custo_log_mult
    config = dict(config)
    config["meta_caixa_90d"] = config["meta_caixa_90d"] * a.meta_mult
    config["orcamento_promocional"] = config["orcamento_promocional"] * a.orcamento_mult
    return catalogo, config


def etapa_brute_force_vs_ga(catalogo, params, k, seed, ga_cfg):
    tab_full = construir_tabela(catalogo, params)
    idx = selecionar_subconjunto(tab_full, k, seed)
    sub = catalogo.iloc[idx].reset_index(drop=True)
    tab = construir_tabela(sub, params)

    (bf_esc, bf_val, combos), t_bf, m_bf = medir(brute_force, tab)
    cfg_sub = ConfigGA(populacao=60, geracoes_max=ga_cfg.geracoes_max, taxa_mutacao=ga_cfg.taxa_mutacao,
                       paciencia=60, seed=seed)
    ga, t_ga, m_ga = medir(algoritmo_genetico, tab, cfg_sub)
    gap = 100 * (bf_val - ga.melhor_fitness) / abs(bf_val)
    rot = [f"{x:.2f}" for x in params.niveis] + ["susp"]
    tabela = pd.DataFrame([
        {"metodo": "Brute Force", "fitness": round(bf_val, 2), "politica": " ".join(rot[i] for i in bf_esc),
         "avaliacoes": combos, "tempo_s": round(t_bf, 4), "pico_mem_MB": round(m_bf, 2), "gap_%": 0.0},
        {"metodo": "GA", "fitness": round(ga.melhor_fitness, 2), "politica": " ".join(rot[i] for i in ga.melhor),
         "avaliacoes": cfg_sub.populacao * (ga.geracoes + 1), "tempo_s": round(t_ga, 4),
         "pico_mem_MB": round(m_ga, 2), "gap_%": round(gap, 4)},
    ])
    return tabela, [tab.skus[i] for i in range(tab.n)], combos


def politica_atual(tab, params) -> np.ndarray:
    """Indivíduo semente: nível 1.00 onde viável, senão o menor preço viável acima, senão suspender."""
    from questao_01.modelo import _opcao_referencia
    return np.array([_opcao_referencia(tab.permitido[i], params) for i in range(tab.n)])


def executar(a) -> dict:
    catalogo, config, manifesto = carregar_q1()
    seed = a.seed if a.seed is not None else manifesto["seed_final"]
    catalogo, config = aplicar_cenario(catalogo, config, a)
    params = parametros_de_config(config, max_frac_desconto=a.max_frac_desconto, fator_giro=a.fator_giro)
    saida = a.saida
    saida.mkdir(parents=True, exist_ok=True)

    # ---------- A) Brute Force valida o GA num subconjunto ----------
    ga_a = ConfigGA(populacao=a.ga_pop, taxa_mutacao=a.ga_mut, geracoes_max=a.ga_geracoes, seed=seed)
    tab_bf, skus_bf, _ = etapa_brute_force_vs_ga(catalogo, params, max(6, a.subconjunto), seed, ga_a)
    tab_bf.to_csv(saida / "q1_bruteforce_vs_ga.csv", index=False)
    print("\n[A] Brute Force x GA no subconjunto", skus_bf)
    print(tab_bf.to_string(index=False))

    # ---------- B) GA no catálogo completo, duas configurações ----------
    tab = construir_tabela(catalogo, params)
    semente = politica_atual(tab, params)
    ga_b = ConfigGA(populacao=max(2 * a.ga_pop, 60), taxa_mutacao=a.ga_mut / 2.5,
                    geracoes_max=a.ga_geracoes, paciencia=100, torneio=4, seed=seed)
    resultados, linhas_cmp = {}, []
    for nome, cfg in (("A", ga_a), ("B", ga_b)):
        r, t, mem = medir(algoritmo_genetico, tab, cfg, semente)
        resultados[nome] = r
        linhas_cmp.append({"config": nome, "populacao": cfg.populacao, "mutacao": cfg.taxa_mutacao,
                           "torneio": cfg.torneio, "elitismo": cfg.elitismo, "paciencia": cfg.paciencia,
                           "geracoes": r.geracoes, "melhor_fitness": round(r.melhor_fitness, 2),
                           "tempo_s": round(t, 4), "pico_mem_MB": round(mem, 3)})
    cmp_ga = pd.DataFrame(linhas_cmp)
    cmp_ga.to_csv(saida / "q1_comparacao_ga.csv", index=False)
    print("\n[B] GA no catálogo completo — duas configurações")
    print(cmp_ga.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nome, r in resultados.items():
        ax.plot(r.historico_melhor, label=f"config {nome} — melhor")
        ax.plot(r.historico_media, "--", alpha=0.6, label=f"config {nome} — média viável")
    ax.set_xlabel("geração")
    ax.set_ylabel("fitness (R$)")
    ax.set_title("Q1 — Fitness por geração do GA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(saida / "q1_fitness_por_geracao.png", dpi=120)
    plt.close(fig)

    melhor_nome = max(resultados, key=lambda k: resultados[k].melhor_fitness)
    escolha = resultados[melhor_nome].melhor

    # ---------- C) Greedy distribui o orçamento sobre a política do GA ----------
    politica = resumo_politica(tab, escolha, catalogo)
    (greedy_df), t_gr, m_gr = medir(alocar_incentivo, tab, escolha, catalogo, config["orcamento_promocional"])
    antes_depois = politica.merge(greedy_df, on="sku")
    antes_depois.to_csv(saida / "q1_politica_final.csv", index=False)

    margem_unit = tab.preco[np.arange(tab.n), escolha] * (1 - catalogo["taxa_marketplace"].to_numpy()) \
        - catalogo["custo_unitario"].to_numpy() - catalogo["custo_logistico"].to_numpy()
    margem_unit = np.where(tab.preco[np.arange(tab.n), escolha] > 0, margem_unit, 0.0)
    un_antes = greedy_df["unidades_antes"].to_numpy()
    un_depois = greedy_df["unidades_depois"].to_numpy()
    caixa_antes = float((un_antes * margem_unit).sum())
    caixa_depois = float((un_depois * margem_unit).sum())
    receita_depois = float((un_depois * greedy_df["preco_ga"].to_numpy()).sum())
    gasto = float(greedy_df["custo_incentivo"].sum())
    estoque_total = float(catalogo["estoque"].sum())

    print("\n[C] Greedy — SKUs que receberam incentivo")
    rec = greedy_df[greedy_df["incentivo_pct"] > 0]
    print(rec.to_string(index=False) if len(rec) else "(nenhum)")

    meta, divida = config["meta_caixa_90d"], config["divida_total_90d"]
    resumo = {
        "grupo": manifesto["grupo"], "id_instancia": manifesto["id_instancia"], "seed": seed,
        "cenario": {"meta_mult": a.meta_mult, "custo_log_mult": a.custo_log_mult,
                    "orcamento_mult": a.orcamento_mult, "skus_removidos": a.remover_sku or []},
        "n_skus": int(tab.n), "skus_suspensos": int((escolha == params.idx_suspender).sum()),
        "skus_com_desconto": int(tab.desconto[escolha].sum()), "limite_descontos": int(np.floor(params.max_frac_desconto * tab.n)),
        "ga_vencedor": melhor_nome,
        "caixa_politica_ga": round(caixa_antes, 2),
        "caixa_apos_greedy": round(caixa_depois, 2),
        "incentivo_gasto": round(gasto, 2),
        "orcamento_promocional": round(config["orcamento_promocional"], 2),
        "uso_orcamento_%": round(100 * gasto / config["orcamento_promocional"], 2) if config["orcamento_promocional"] else 0.0,
        "caixa_liquido_do_incentivo": round(caixa_depois - gasto, 2),
        "margem_operacional_%": round(100 * caixa_depois / receita_depois, 2) if receita_depois else 0.0,
        "receita": round(receita_depois, 2),
        "estoque_inicial_un": estoque_total,
        "estoque_remanescente_un": round(estoque_total - float(un_depois.sum()), 1),
        "estoque_remanescente_%": round(100 * (1 - float(un_depois.sum()) / estoque_total), 2),
        "meta_caixa_90d": round(meta, 2),
        "meta_atingida": bool(caixa_depois >= meta),
        "folga_ou_insuficiencia": round(caixa_depois - meta, 2),
        "divida_total_90d": round(divida, 2),
        "cobertura_divida_%": round(100 * caixa_depois / divida, 2),
        "tempo_greedy_s": round(t_gr, 5), "pico_mem_greedy_MB": round(m_gr, 3),
    }
    salvar_json(resumo, saida / "q1_resumo.json")
    print("\n[Resultado final]")
    for k, v in resumo.items():
        print(f"  {k}: {v}")

    if a.modo == "completo":
        desempenho(catalogo, params, seed, saida)
    return resumo


def desempenho(catalogo, params, seed, saida):
    """Escalabilidade (entradas pequena/média/grande) e otimização antes/depois."""
    linhas = []
    tab_full = construir_tabela(catalogo, params)
    for k in (5, 6, 7, 8):
        idx = selecionar_subconjunto(tab_full, k, seed)
        tab = construir_tabela(catalogo.iloc[idx].reset_index(drop=True), params)
        _, t, m = medir(brute_force, tab)
        linhas.append({"algoritmo": "Brute Force", "n_skus": k, "combinacoes": numero_combinacoes(tab),
                       "tempo_s": round(t, 4), "pico_mem_MB": round(m, 2)})
    for n in (10, 20, len(catalogo)):
        tab = construir_tabela(catalogo.iloc[:n].reset_index(drop=True), params)
        cfg = ConfigGA(geracoes_max=150, paciencia=10**9, seed=seed)  # nº fixo de gerações
        _, t, m = medir(algoritmo_genetico, tab, cfg)
        linhas.append({"algoritmo": "GA (150 ger, P=120)", "n_skus": n, "combinacoes": "",
                       "tempo_s": round(t, 4), "pico_mem_MB": round(m, 2)})
        _, t, m = medir(alocar_incentivo, tab, politica_atual(tab, params),
                        catalogo.iloc[:n].reset_index(drop=True), 1e5)
        linhas.append({"algoritmo": "Greedy", "n_skus": n, "combinacoes": "",
                       "tempo_s": round(t, 5), "pico_mem_MB": round(m, 3)})
    esc = pd.DataFrame(linhas)
    esc.to_csv(saida / "q1_escalabilidade.csv", index=False)
    print("\n[Desempenho] Escalabilidade")
    print(esc.to_string(index=False))

    # otimização: fitness em laço Python (recalcula tudo) x tabela pré-computada vetorizada
    pop = np.random.default_rng(seed).integers(0, params.n_opcoes, size=(2000, tab_full.n))
    v_loop, t_loop, m_loop = medir(avaliar_populacao_loop, tab_full, pop, catalogo)
    v_vec, t_vec, m_vec = medir(avaliar_populacao, tab_full, pop)
    ok = np.isfinite(v_vec)
    otm = pd.DataFrame([
        {"versao": "antes: laço Python", "tempo_s": round(t_loop, 4), "pico_mem_MB": round(m_loop, 2)},
        {"versao": "depois: tabela + numpy", "tempo_s": round(t_vec, 5), "pico_mem_MB": round(m_vec, 2)},
    ])
    otm["speedup"] = [1.0, round(t_loop / t_vec, 1)]
    otm["resultados_identicos"] = bool(np.allclose(v_loop[ok], v_vec[ok]) and (np.isfinite(v_loop) == ok).all())
    otm.to_csv(saida / "q1_otimizacao_antes_depois.csv", index=False)
    print("\n[Desempenho] Otimização da função de fitness (2000 indivíduos)")
    print(otm.to_string(index=False))


if __name__ == "__main__":
    executar(argumentos())

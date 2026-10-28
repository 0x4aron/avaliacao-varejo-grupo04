"""Questão 2 — Plano de Sobrevivência em 12 Semanas.
Fluxo: Monte Carlo (construção) -> Programação Dinâmica -> Greedy -> Monte Carlo (validação).

Exemplos:
    python questao_02/main.py --cenarios 5000
    python questao_02/main.py --concorrente-mult 0.92          # concorrente -8%
    python questao_02/main.py --custo-log-mult 1.15 --orcamento-mult 0.75
    python questao_02/main.py --divida-mult 1.10 --remover-sku Q2-003
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

from comum.dados import carregar_q2, remover_skus, salvar_json  # noqa: E402
from comum.medicao import medir  # noqa: E402
from questao_02.dynamic_programming import dp_sku, trajetoria_esperada  # noqa: E402
from questao_02.greedy import ConfigDestaque, alocar_destaques  # noqa: E402
from questao_02.monte_carlo import estatisticas_construcao, parametros_de_config, validar_plano  # noqa: E402

PASTA_RES = Path(__file__).resolve().parent / "resultados"
OFFSET_CONSTRUCAO, OFFSET_VALIDACAO = 1, 2       # seeds diferentes => cenários independentes


def argumentos(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None, help="padrão: seed_final do manifesto")
    ap.add_argument("--cenarios", type=int, default=5000, help="cenários da validação final (>= 5000)")
    ap.add_argument("--cenarios-construcao", type=int, default=None,
                    help="padrão: n_simulacoes_recomendadas do config (2000)")
    ap.add_argument("--modo", choices=["rapido", "completo"], default="rapido")
    ap.add_argument("--divida-mult", type=float, default=1.0, help="cenário: dívidas/meta x mult")
    ap.add_argument("--concorrente-mult", type=float, default=1.0)
    ap.add_argument("--custo-log-mult", type=float, default=1.0)
    ap.add_argument("--orcamento-mult", type=float, default=1.0)
    ap.add_argument("--remover-sku", nargs="*", default=None)
    ap.add_argument("--urgencia", type=float, default=0.0, help="desconto semanal da DP (antecipa caixa)")
    ap.add_argument("--max-destaques", type=int, default=4)
    ap.add_argument("--uplift", type=float, default=0.20)
    ap.add_argument("--percentil-risco", type=float, default=5.0)
    ap.add_argument("--saida", type=Path, default=PASTA_RES)
    return ap.parse_args(argv)


def aplicar_cenario(catalogo, config, a):
    catalogo = remover_skus(catalogo, a.remover_sku)
    config = dict(config)
    config["dividas"] = {k: v * a.divida_mult for k, v in config["dividas"].items()}
    config["orcamento_promocional_semanal"] = config["orcamento_promocional_semanal"] * a.orcamento_mult
    return catalogo, config


def construir_plano(catalogo, config, params, stats, urgencia, cfg_dest):
    """DP para cada SKU (consome a tabela do MC) + Greedy (consome a trajetória da DP)."""
    T = int(config["numero_semanas"])
    politicas, niveis, estoques = [], [], []
    estados = consultas = 0
    for i, s0 in enumerate(catalogo["estoque_inicial"].to_numpy()):
        res = dp_sku(int(s0), T, stats["demanda_quantis"][i], stats["margem_unit_media"][i],
                     urgencia=urgencia, permitido=stats["margem_unit_media"][i] > 0)
        politicas.append(res.politica)
        nv, es = trajetoria_esperada(res, int(s0), stats["demanda_media"][i])
        niveis.append(nv)
        estoques.append(es)
        estados += res.estados
        consultas += res.consultas
    niveis, estoques = np.array(niveis), np.array(estoques)
    dest, custo, decisoes = alocar_destaques(niveis, estoques, stats["preco"], stats["demanda_media"],
                                             stats["margem_unit_media"], config["orcamento_promocional_semanal"],
                                             cfg_dest)
    return {"politicas": politicas, "niveis": niveis, "estoques": estoques, "destaques": dest,
            "custo_destaque": custo, "decisoes": decisoes, "estados": estados, "consultas": consultas}


def politica_fixa(catalogo, T, params):
    """Comparação: preço atual (nível 1.00) todas as semanas, sem destaques."""
    j = int(np.argmin(np.abs(params.niveis - 1.0)))
    return [np.full((T, int(s) + 1), j) for s in catalogo["estoque_inicial"]]


def resumir_validacao(v, config, percentil):
    nomes = sorted(config["dividas"], key=lambda k: int(k.split("_")[1]))
    out = {f"P_pagar_{n}": round(float(v.pagou[:, k].mean()), 4) for k, n in enumerate(nomes)}
    out["P_pagar_todas"] = round(float(v.pagou.all(axis=1).mean()), 4)
    out["caixa_final_esperado"] = round(float(v.caixa_final.mean()), 2)
    out[f"caixa_final_P{percentil:g}"] = round(float(np.percentile(v.caixa_final, percentil)), 2)
    out["caixa_final_desvio"] = round(float(v.caixa_final.std()), 2)
    out["margem_operacional_%"] = round(100 * float(v.contribuicao.sum() / v.receita.sum()), 2)
    out["estoque_final_medio_un"] = round(float(v.estoque_final.sum(axis=1).mean()), 1)
    out["cenarios_com_falha"] = int((~v.pagou.all(axis=1)).sum())
    return out


def executar(a) -> dict:
    catalogo, config, manifesto = carregar_q2()
    seed = a.seed if a.seed is not None else manifesto["seed_final"]
    catalogo, config = aplicar_cenario(catalogo, config, a)
    params = parametros_de_config(config, mult_concorrente=a.concorrente_mult, mult_custo_log=a.custo_log_mult)
    n_constr = a.cenarios_construcao or int(config["n_simulacoes_recomendadas"])
    T = int(config["numero_semanas"])
    cfg_dest = ConfigDestaque(max_destaques=a.max_destaques, uplift=a.uplift)
    saida = a.saida
    saida.mkdir(parents=True, exist_ok=True)

    # ---------- A) Monte Carlo de construção ----------
    stats, t_mc, m_mc = medir(estatisticas_construcao, catalogo, params, n_constr, seed + OFFSET_CONSTRUCAO)
    rot = [f"{x:.2f}" for x in params.niveis]
    tab_mc = pd.DataFrame([
        {"sku": catalogo["sku"].iloc[i], "nivel": rot[j], "preco": round(stats["preco"][i, j], 2),
         "demanda_media_sem": round(stats["demanda_media"][i, j], 2),
         "quantis_demanda": " ".join(map(str, stats["demanda_quantis"][i, j])),
         "margem_unit_media": round(stats["margem_unit_media"][i, j], 2)}
        for i in range(len(catalogo)) for j in range(len(params.niveis))])
    tab_mc.to_csv(saida / "q2_estatisticas_mc.csv", index=False)
    print(f"[A] Monte Carlo de construção: {n_constr} cenários, {t_mc:.3f}s, pico {m_mc:.2f} MB")

    # ---------- B) DP + C) Greedy ----------
    plano, t_plano, m_plano = medir(construir_plano, catalogo, config, params, stats, a.urgencia, cfg_dest)
    print(f"[B] DP: {plano['estados']} estados resolvidos, {plano['consultas']} consultas a V[t+1] "
          f"(reuso médio {plano['consultas'] / plano['estados']:.0f}x por estado)")
    print(f"[C] Greedy: {int(plano['destaques'].sum())} destaques em {T} semanas, "
          f"gasto total R$ {plano['custo_destaque'].sum():,.2f}")

    rot_ext = rot + ["susp"]
    traj = pd.DataFrame([
        {"sku": catalogo["sku"].iloc[i], "semana": t + 1, "nivel": rot_ext[plano["niveis"][i, t]],
         "preco": round(float(stats["preco"][i, plano["niveis"][i, t]]) if plano["niveis"][i, t] < len(rot) else 0.0, 2),
         "estoque_inicio_semana": round(float(plano["estoques"][i, t]), 1),
         "destaque": bool(plano["destaques"][t, i])}
        for i in range(len(catalogo)) for t in range(T)])
    traj.to_csv(saida / "q2_trajetoria_dp.csv", index=False)
    dec = pd.DataFrame(plano["decisoes"])
    if len(dec):
        dec.insert(1, "sku", catalogo["sku"].to_numpy()[dec["sku_idx"]])
        dec = dec.drop(columns="sku_idx")
    dec.to_csv(saida / "q2_destaques_greedy.csv", index=False)

    # ---------- D) Monte Carlo de validação (cenários novos) ----------
    v, t_val, m_val = medir(validar_plano, catalogo, config, params, plano["politicas"], plano["destaques"],
                            plano["custo_destaque"], cfg_dest.uplift, a.cenarios, seed + OFFSET_VALIDACAO)
    v_fix = validar_plano(catalogo, config, params, politica_fixa(catalogo, T, params),
                          np.zeros((T, len(catalogo)), bool), np.zeros((T, len(catalogo))), 0.0,
                          a.cenarios, seed + OFFSET_VALIDACAO)
    res_plano = resumir_validacao(v, config, a.percentil_risco)
    res_fixo = resumir_validacao(v_fix, config, a.percentil_risco)
    comparacao = pd.DataFrame({"plano_MC_DP_Greedy": res_plano, "preco_fixo_1.00": res_fixo})
    comparacao.to_csv(saida / "q2_validacao.csv")
    print(f"\n[D] Validação em {a.cenarios} cenários independentes ({t_val:.2f}s, pico {m_val:.1f} MB)")
    print(comparacao.to_string())

    graficos(v, v_fix, traj, catalogo, config, saida)

    resumo = {"grupo": manifesto["grupo"], "id_instancia": manifesto["id_instancia"], "seed": seed,
              "cenario": {"divida_mult": a.divida_mult, "concorrente_mult": a.concorrente_mult,
                          "custo_log_mult": a.custo_log_mult, "orcamento_mult": a.orcamento_mult,
                          "skus_removidos": a.remover_sku or [], "urgencia": a.urgencia},
              "n_skus": len(catalogo), "cenarios_construcao": n_constr, "cenarios_validacao": a.cenarios,
              "dividas": config["dividas"], "caixa_inicial": config["caixa_inicial"],
              "limite_risco_falha": config["limite_risco_falha"],
              "dentro_do_limite_de_risco": bool(1 - res_plano["P_pagar_todas"] <= config["limite_risco_falha"]),
              "plano": res_plano, "preco_fixo": res_fixo,
              "dp_estados": plano["estados"], "dp_consultas": plano["consultas"],
              "tempos_s": {"mc_construcao": round(t_mc, 4), "dp_mais_greedy": round(t_plano, 4),
                           "mc_validacao": round(t_val, 4)},
              "pico_mem_MB": {"mc_construcao": round(m_mc, 2), "dp_mais_greedy": round(m_plano, 2),
                              "mc_validacao": round(m_val, 2)}}
    salvar_json(resumo, saida / "q2_resumo.json")

    if a.modo == "completo":
        estresse(catalogo, config, params, stats, plano, a, seed, saida)
        escalabilidade(catalogo, config, params, a, seed, saida)
    return resumo


def graficos(v, v_fix, traj, catalogo, config, saida):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(v.caixa_final / 1e3, bins=60, alpha=0.75, label="plano MC→DP→Greedy")
    ax.hist(v_fix.caixa_final / 1e3, bins=60, alpha=0.6, label="preço fixo 1.00")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("caixa final após as 3 dívidas (R$ mil)")
    ax.set_ylabel("cenários")
    ax.set_title("Q2 — Distribuição do caixa final (validação)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(saida / "q2_distribuicao_caixa_final.png", dpi=120)
    plt.close(fig)

    # produtos representativos: maior estoque, maior preço e mais destaques
    grupo = traj.groupby("sku")
    reps = list(dict.fromkeys([
        catalogo.loc[catalogo["estoque_inicial"].idxmax(), "sku"],
        catalogo.loc[catalogo["preco_atual"].idxmax(), "sku"],
        grupo["destaque"].sum().idxmax(),
    ]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for sku in reps:
        d = traj[traj["sku"] == sku]
        axes[0].step(d["semana"], d["preco"], where="post", label=sku)
        axes[1].plot(d["semana"], d["estoque_inicio_semana"], marker="o", label=sku)
    for k in config["dividas"]:
        for ax in axes:
            ax.axvline(int(k.split("_")[1]), color="gray", ls=":", lw=0.8)
    axes[0].set_ylabel("preço (R$)")
    axes[1].set_ylabel("estoque esperado")
    axes[1].set_xlabel("semana")
    axes[0].set_title("Q2 — Trajetória de preço e estoque (DP) — produtos representativos")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(saida / "q2_trajetoria_preco_estoque.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    p10, p50, p90 = np.percentile(v.caixa_semanal / 1e3, [10, 50, 90], axis=0)
    sem = np.arange(1, len(p50) + 1)
    ax.fill_between(sem, p10, p90, alpha=0.3, label="P10–P90")
    ax.plot(sem, p50, label="mediana")
    ax.set_xlabel("semana (após pagamento da dívida, quando houver)")
    ax.set_ylabel("caixa (R$ mil)")
    ax.set_title("Q2 — Caixa semanal simulado")
    ax.legend()
    fig.tight_layout()
    fig.savefig(saida / "q2_caixa_semanal.png", dpi=120)
    plt.close(fig)


def estresse(catalogo, config, params, stats, plano, a, seed, saida):
    """Onde a falha começa: dívidas maiores e choques adversos, plano x preço fixo."""
    T = int(config["numero_semanas"])
    linhas = []
    for mult in (1.0, 1.2, 1.4, 1.5, 1.6, 1.8):
        cfg = dict(config)
        cfg["dividas"] = {k: v * mult for k, v in config["dividas"].items()}
        for nome, pol, dest, custo, upl in (
            ("plano", plano["politicas"], plano["destaques"], plano["custo_destaque"], a.uplift),
            ("fixo_1.00", politica_fixa(catalogo, T, params), np.zeros((T, len(catalogo)), bool),
             np.zeros((T, len(catalogo))), 0.0)):
            v = validar_plano(catalogo, cfg, params, pol, dest, custo, upl, 2000, seed + OFFSET_VALIDACAO)
            r = resumir_validacao(v, cfg, a.percentil_risco)
            linhas.append({"divida_mult": mult, "politica": nome, **{k: r[k] for k in (
                "P_pagar_semana_4", "P_pagar_semana_8", "P_pagar_semana_12", "P_pagar_todas",
                "caixa_final_esperado")}})
    est = pd.DataFrame(linhas)
    est.to_csv(saida / "q2_estresse_divida.csv", index=False)
    print("\n[Estresse] Multiplicador das dívidas (2000 cenários cada)")
    print(est.to_string(index=False))


def escalabilidade(catalogo, config, params, a, seed, saida):
    """Três tamanhos de entrada: nº de SKUs e escala do estoque (S domina a DP)."""
    T = int(config["numero_semanas"])
    linhas = []
    for n_sku, esc in ((8, 1), (16, 1), (24, 1), (24, 2), (24, 4)):
        cat = catalogo.iloc[:n_sku].copy().reset_index(drop=True)
        cat["estoque_inicial"] = cat["estoque_inicial"] * esc
        cat["demanda_base_semanal"] = cat["demanda_base_semanal"] * esc
        st, t_mc, m_mc = medir(estatisticas_construcao, cat, params, 2000, seed + OFFSET_CONSTRUCAO)
        pl, t_dp, m_dp = medir(construir_plano, cat, config, params, st, 0.0, ConfigDestaque())
        _, t_gr, m_gr = medir(alocar_destaques, pl["niveis"], pl["estoques"], st["preco"], st["demanda_media"],
                              st["margem_unit_media"], config["orcamento_promocional_semanal"], ConfigDestaque())
        _, t_v, m_v = medir(validar_plano, cat, config, params, pl["politicas"], pl["destaques"],
                            pl["custo_destaque"], 0.2, 5000, seed + OFFSET_VALIDACAO)
        linhas.append({"n_skus": n_sku, "escala_estoque": esc, "soma_estoque": int(cat["estoque_inicial"].sum()),
                       "estados_dp": pl["estados"],
                       "mc_constr_s": round(t_mc, 3), "mc_constr_MB": round(m_mc, 2),
                       "dp_greedy_s": round(t_dp, 3), "dp_greedy_MB": round(m_dp, 2),
                       "greedy_s": round(t_gr, 4), "greedy_MB": round(m_gr, 3),
                       "mc_valid_s": round(t_v, 3), "mc_valid_MB": round(m_v, 2)})
    esc_df = pd.DataFrame(linhas)
    esc_df.to_csv(saida / "q2_escalabilidade.csv", index=False)
    print("\n[Desempenho] Escalabilidade")
    print(esc_df.to_string(index=False))


if __name__ == "__main__":
    executar(argumentos())

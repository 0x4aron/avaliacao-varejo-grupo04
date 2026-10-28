"""Gera docs/arquitetura.png (fluxo de dados entre os algoritmos das duas questões)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def caixa(ax, x, y, texto, cor):
    ax.add_patch(FancyBboxPatch((x - 1.15, y - 0.45), 2.3, 0.9, boxstyle="round,pad=0.05",
                                fc=cor, ec="#333", lw=1.2))
    ax.text(x, y, texto, ha="center", va="center", fontsize=8.5)


def seta(ax, a, b, rotulo=""):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.2, color="#333"))
    if rotulo:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.22, rotulo, ha="center", fontsize=7, color="#444")


fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.4)
ax.axis("off")

ax.text(0.2, 6.0, "Questão 1 — Operação Caixa 90", fontsize=11, weight="bold")
caixa(ax, 1.5, 4.9, "data/q1_*.csv/json\n(hash SHA-256 verificado)", "#eeeeee")
caixa(ax, 4.6, 4.9, "modelo.construir_tabela\ncaixa[n, L+1]  pré-computado", "#dde8f5")
caixa(ax, 7.8, 5.5, "A) Brute Force\nsubconjunto k=7 (ótimo exato)", "#f9e0c7")
caixa(ax, 7.8, 4.3, "B) Algoritmo Genético\ncatálogo completo (n=30)", "#d5ecd4")
caixa(ax, 11.2, 4.3, "C) Greedy (heap)\norçamento de incentivo", "#f3d6e4")
seta(ax, (2.65, 4.9), (3.45, 4.9))
seta(ax, (5.75, 5.05), (6.65, 5.45))
seta(ax, (5.75, 4.75), (6.65, 4.35))
seta(ax, (7.8, 5.05), (7.8, 4.75), "")
ax.text(8.0, 4.85, "gap % valida o GA", fontsize=7, color="#444")
seta(ax, (8.95, 4.3), (10.05, 4.3), "vetor de preços")

ax.text(0.2, 2.9, "Questão 2 — Sobrevivência em 12 semanas", fontsize=11, weight="bold")
caixa(ax, 1.5, 1.6, "data/q2_*.csv/json\n3 fontes de incerteza", "#eeeeee")
caixa(ax, 4.2, 1.6, "A) Monte Carlo construção\n2000 cenários → quantis", "#dde8f5")
caixa(ax, 6.9, 1.6, "B) DP bottom-up\nV[t, s]  semana × estoque", "#d5ecd4")
caixa(ax, 9.6, 1.6, "C) Greedy semanal\nvagas de destaque", "#f3d6e4")
caixa(ax, 9.6, 0.35, "D) Monte Carlo validação\n5000 cenários novos", "#f9e0c7")
seta(ax, (2.65, 1.6), (3.05, 1.6))
seta(ax, (5.35, 1.6), (5.75, 1.6), "demanda/margem")
seta(ax, (8.05, 1.6), (8.45, 1.6), "trajetória")
seta(ax, (9.6, 1.15), (9.6, 0.8), "")
seta(ax, (6.9, 1.15), (8.45, 0.45), "política[t, s]")
ax.text(10.85, 0.35, "P(pagar sem. 4, 8, 12)\ncaixa final, P5", fontsize=7.5, va="center")

fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "arquitetura.png", dpi=130)

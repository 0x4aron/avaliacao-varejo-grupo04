"""Leitura da instância oficial, verificação de integridade e cenários-surpresa.

Os arquivos em data/ NUNCA são reescritos. Toda alteração de cenário é aplicada
em memória sobre o DataFrame / dicionário carregado.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "data"

COLUNAS_Q1 = [
    "sku", "categoria", "custo_unitario", "preco_atual", "estoque",
    "demanda_base_90d", "elasticidade", "taxa_marketplace",
    "custo_logistico", "idade_estoque_dias",
]
COLUNAS_Q2 = [
    "sku", "categoria", "custo_unitario", "preco_atual", "estoque_inicial",
    "demanda_base_semanal", "elasticidade", "taxa_marketplace",
    "custo_logistico_base", "preco_concorrente_base",
]


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar_manifesto(pasta: Path = PASTA_DADOS) -> dict:
    with open(pasta / "manifesto_dados.json", encoding="utf-8") as f:
        return json.load(f)


def verificar_integridade(pasta: Path = PASTA_DADOS) -> dict:
    """Recalcula o SHA-256 de cada arquivo e compara com o manifesto oficial."""
    manifesto = carregar_manifesto(pasta)
    for nome, esperado in manifesto["arquivos"].items():
        obtido = sha256_arquivo(pasta / nome)
        if obtido != esperado:
            raise ValueError(f"Hash divergente em {nome}: arquivo alterado após a geração.")
    return manifesto


def _ler_csv(caminho: Path, colunas: list[str]) -> pd.DataFrame:
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    if list(df.columns) != colunas:
        raise ValueError(f"Colunas inesperadas em {caminho.name}: {list(df.columns)}")
    return df


def carregar_q1(pasta: Path = PASTA_DADOS) -> tuple[pd.DataFrame, dict, dict]:
    manifesto = verificar_integridade(pasta)
    g = manifesto["grupo"]
    catalogo = _ler_csv(pasta / f"q1_catalogo_{g}.csv", COLUNAS_Q1)
    with open(pasta / f"q1_config_{g}.json", encoding="utf-8") as f:
        config = json.load(f)
    return catalogo, config, manifesto


def carregar_q2(pasta: Path = PASTA_DADOS) -> tuple[pd.DataFrame, dict, dict]:
    manifesto = verificar_integridade(pasta)
    g = manifesto["grupo"]
    catalogo = _ler_csv(pasta / f"q2_catalogo_{g}.csv", COLUNAS_Q2)
    with open(pasta / f"q2_config_{g}.json", encoding="utf-8") as f:
        config = json.load(f)
    return catalogo, config, manifesto


def remover_skus(catalogo: pd.DataFrame, skus: list[str] | None) -> pd.DataFrame:
    """Cenário-surpresa 'um SKU deixa de estar disponível'."""
    if not skus:
        return catalogo
    faltando = set(skus) - set(catalogo["sku"])
    if faltando:
        raise ValueError(f"SKU(s) inexistente(s): {sorted(faltando)}")
    return catalogo[~catalogo["sku"].isin(skus)].reset_index(drop=True)


def salvar_json(obj: dict, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)

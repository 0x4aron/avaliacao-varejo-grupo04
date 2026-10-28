import numpy as np
import pandas as pd
import pytest

from comum.dados import carregar_q1, remover_skus
from questao_01.brute_force import brute_force, numero_combinacoes
from questao_01.genetic_algorithm import ConfigGA, algoritmo_genetico
from questao_01.greedy import alocar_incentivo
from questao_01.modelo import (avaliar, avaliar_populacao, avaliar_populacao_loop, construir_tabela,
                               parametros_de_config)


@pytest.fixture(scope="module")
def instancia():
    cat, cfg, man = carregar_q1()
    return cat, cfg, parametros_de_config(cfg)


def catalogo_pequeno():
    return pd.DataFrame({
        "sku": ["A", "B", "C"], "categoria": ["x"] * 3,
        "custo_unitario": [50.0, 80.0, 500.0], "preco_atual": [100.9, 150.9, 400.9],
        "estoque": [100, 60, 10], "demanda_base_90d": [80, 70, 8],
        "elasticidade": [1.5, 1.2, 1.0], "taxa_marketplace": [0.1, 0.1, 0.1],
        "custo_logistico": [5.0, 10.0, 20.0], "idade_estoque_dias": [200, 30, 100],
    })


def test_caixa_base_reproduz_gerador(instancia):
    cat, cfg, _ = instancia
    u = np.minimum(cat["estoque"], cat["demanda_base_90d"])
    m = (cat["preco_atual"] * (1 - cat["taxa_marketplace"]) - cat["custo_unitario"] - cat["custo_logistico"]).clip(lower=0)
    base = float((u * m).sum())
    assert 0.82 <= cfg["meta_caixa_90d"] / base <= 0.96


def test_sku_sem_preco_viavel_so_pode_suspender():
    params = parametros_de_config({"margem_minima_percentual": 0.05, "niveis_preco_relativos": [0.9, 1.0, 1.1]})
    tab = construir_tabela(catalogo_pequeno(), params)
    # SKU C custa mais do que vale: só a opção "suspender" é permitida
    assert tab.permitido[2].tolist() == [False, False, False, True]
    assert tab.permitido[0].all()


def test_brute_force_acha_otimo_conhecido():
    params = parametros_de_config({"margem_minima_percentual": 0.05, "niveis_preco_relativos": [0.9, 1.0, 1.1]},
                                  max_frac_desconto=1.0, fator_giro=0.0)
    tab = construir_tabela(catalogo_pequeno(), params)
    esc, val, n = brute_force(tab)
    assert n == numero_combinacoes(tab) == 4 * 4 * 1
    # sem restrições acopladas o ótimo é o argmax independente de cada SKU
    esperado = np.array([np.argmax(tab.fitness[i]) for i in range(tab.n)])
    assert np.array_equal(esc, esperado)
    assert val == pytest.approx(avaliar(tab, esperado))


def test_ga_iguala_brute_force_no_subconjunto(instancia):
    cat, cfg, params = instancia
    tab = construir_tabela(cat.iloc[[0, 1, 4, 5, 7, 8]].reset_index(drop=True), params)
    _, bf_val, _ = brute_force(tab)
    ga = algoritmo_genetico(tab, ConfigGA(populacao=60, seed=1))
    assert ga.melhor_fitness == pytest.approx(bf_val)


def test_ga_reprodutivel_por_seed(instancia):
    cat, cfg, params = instancia
    tab = construir_tabela(cat, params)
    r1 = algoritmo_genetico(tab, ConfigGA(geracoes_max=30, seed=7))
    r2 = algoritmo_genetico(tab, ConfigGA(geracoes_max=30, seed=7))
    assert np.array_equal(r1.melhor, r2.melhor) and r1.historico_melhor == r2.historico_melhor


def test_ga_so_gera_genes_permitidos(instancia):
    cat, cfg, params = instancia
    tab = construir_tabela(cat, params)
    r = algoritmo_genetico(tab, ConfigGA(geracoes_max=20, seed=3))
    assert tab.permitido[np.arange(tab.n), r.melhor].all()


def test_penalidade_limite_de_descontos():
    params = parametros_de_config({"margem_minima_percentual": 0.0, "niveis_preco_relativos": [0.9, 1.0]},
                                  max_frac_desconto=0.0, fator_giro=0.0)
    tab = construir_tabela(catalogo_pequeno().iloc[:2], params)
    sem_desc, com_desc = np.array([1, 1]), np.array([0, 1])
    assert avaliar(tab, com_desc) < avaliar(tab, sem_desc) - 1e5


def test_fitness_vetorizado_igual_ao_laco(instancia):
    cat, cfg, params = instancia
    tab = construir_tabela(cat, params)
    pop = np.random.default_rng(0).integers(0, params.n_opcoes, size=(100, tab.n))
    a, b = avaliar_populacao(tab, pop), avaliar_populacao_loop(tab, pop, cat)
    ok = np.isfinite(a)
    assert (np.isfinite(b) == ok).all() and np.allclose(a[ok], b[ok])


def test_greedy_respeita_orcamento(instancia):
    cat, cfg, params = instancia
    tab = construir_tabela(cat, params)
    esc = np.array([np.flatnonzero(tab.permitido[i])[0] for i in range(tab.n)])
    for orc in (0.0, 5000.0, cfg["orcamento_promocional"]):
        g = alocar_incentivo(tab, esc, cat, orc)
        assert g["custo_incentivo"].sum() <= orc + 1e-6
    assert alocar_incentivo(tab, esc, cat, 0.0)["incentivo_pct"].sum() == 0


def test_remover_sku_invalido_gera_erro(instancia):
    cat, _, _ = instancia
    with pytest.raises(ValueError):
        remover_skus(cat, ["NAO-EXISTE"])
    assert len(remover_skus(cat, ["Q1-001"])) == len(cat) - 1

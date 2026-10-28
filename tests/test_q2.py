import numpy as np
import pytest

from comum.dados import carregar_q2
from questao_02.dynamic_programming import dp_sku, trajetoria_esperada
from questao_02.greedy import ConfigDestaque, alocar_destaques
from questao_02.monte_carlo import estatisticas_construcao, parametros_de_config, validar_plano


@pytest.fixture(scope="module")
def instancia():
    cat, cfg, _ = carregar_q2()
    return cat, cfg, parametros_de_config(cfg)


def dp_forca_bruta(S, T, quantis, margem):
    """Recursão direta (sem tabela) para conferir a DP em instâncias minúsculas."""
    L, K = quantis.shape

    def V(t, s):
        if t == T:
            return 0.0
        melhor = V(t + 1, s)  # suspender
        for j in range(L):
            tot = 0.0
            for d in quantis[j]:
                v = min(s, d)
                tot += v * margem[j] + V(t + 1, s - v)
            melhor = max(melhor, tot / K)
        return melhor
    return V(0, S)


def test_dp_igual_recursao_direta():
    quantis = np.array([[1, 2, 3], [0, 1, 2]])
    margem = np.array([5.0, 9.0])
    res = dp_sku(6, 4, quantis, margem)
    assert res.V[0, 6] == pytest.approx(dp_forca_bruta(6, 4, quantis, margem))


def test_dp_estoque_zero_vale_zero():
    res = dp_sku(0, 3, np.array([[1, 2]]), np.array([10.0]))
    assert res.V[0, 0] == 0.0


def test_dp_nao_vende_com_margem_negativa():
    res = dp_sku(10, 3, np.array([[2, 3]]), np.array([-5.0]), permitido=np.array([False]))
    assert (res.politica == 1).all()        # 1 = suspender (L)
    assert res.V[0, 10] == 0.0


def test_trajetoria_reconstruida_tem_T_semanas():
    res = dp_sku(20, 5, np.array([[2, 4], [1, 2]]), np.array([3.0, 6.0]))
    niveis, est = trajetoria_esperada(res, 20, np.array([3.0, 1.5]))
    assert len(niveis) == len(est) == 5 and est[0] == 20
    assert all(a >= b for a, b in zip(est, est[1:]))


def test_monte_carlo_reprodutivel(instancia):
    cat, cfg, p = instancia
    a = estatisticas_construcao(cat, p, 500, 123)
    b = estatisticas_construcao(cat, p, 500, 123)
    c = estatisticas_construcao(cat, p, 500, 124)
    assert np.array_equal(a["demanda_quantis"], b["demanda_quantis"])
    assert not np.array_equal(a["demanda_media"], c["demanda_media"])


def test_preco_maior_reduz_demanda(instancia):
    cat, cfg, p = instancia
    st = estatisticas_construcao(cat, p, 2000, 1)
    assert (np.diff(st["demanda_media"], axis=1) < 0).all()


def test_concorrente_mais_barato_reduz_demanda(instancia):
    cat, cfg, _ = instancia
    base = estatisticas_construcao(cat, parametros_de_config(cfg), 2000, 1)
    menor = estatisticas_construcao(cat, parametros_de_config(cfg, mult_concorrente=0.92), 2000, 1)
    assert (menor["demanda_media"] < base["demanda_media"]).all()


def test_greedy_respeita_vagas_e_orcamento(instancia):
    cat, cfg, p = instancia
    st = estatisticas_construcao(cat, p, 500, 1)
    n, T = len(cat), 12
    niveis = np.full((n, T), 3)
    est = np.tile(cat["estoque_inicial"].to_numpy()[:, None], (1, T)).astype(float)
    cfgd = ConfigDestaque(max_destaques=2)
    dest, custo, _ = alocar_destaques(niveis, est, st["preco"], st["demanda_media"], st["margem_unit_media"],
                                      1000.0, cfgd)
    assert (dest.sum(axis=1) <= 2).all()
    assert (custo.sum(axis=1) <= 1000.0 + 1e-6).all()


def test_validacao_estoque_nunca_negativo_e_divida_impagavel(instancia):
    cat, cfg, p = instancia
    T = 12
    pols = [np.full((T, int(s) + 1), 3) for s in cat["estoque_inicial"]]
    zeros = np.zeros((T, len(cat)))
    v = validar_plano(cat, cfg, p, pols, zeros.astype(bool), zeros, 0.0, 300, 9)
    assert (v.estoque_final >= 0).all()
    cfg_imp = dict(cfg, dividas={k: x * 100 for k, x in cfg["dividas"].items()})
    v2 = validar_plano(cat, cfg_imp, p, pols, zeros.astype(bool), zeros, 0.0, 300, 9)
    assert not v2.pagou.any()

import shutil

import pytest

from comum.dados import PASTA_DADOS, carregar_manifesto, verificar_integridade


def test_manifesto_bate_com_arquivos():
    m = verificar_integridade()
    assert m["grupo"] == "G04" and m["seed_final"] == 1000 * 4 + m["seed_base"]


def test_arquivo_alterado_e_detectado(tmp_path):
    for f in PASTA_DADOS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    nome = next(iter(carregar_manifesto(tmp_path)["arquivos"]))
    with open(tmp_path / nome, "a", encoding="utf-8") as fh:
        fh.write(" ")
    with pytest.raises(ValueError, match="Hash divergente"):
        verificar_integridade(tmp_path)

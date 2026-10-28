#!/usr/bin/env bash
# Reproduz todos os resultados principais: testes, Questão 1 e Questão 2.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
"$PY" --version
"$PY" -m pytest -q
"$PY" questao_01/main.py --modo completo
"$PY" questao_02/main.py --cenarios 5000 --modo completo

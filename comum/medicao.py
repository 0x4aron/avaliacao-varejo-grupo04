"""Medição real de tempo (time.perf_counter) e memória de pico (tracemalloc)."""
from __future__ import annotations

import time
import tracemalloc


def medir(func, *args, **kwargs):
    """Executa func e retorna (resultado, segundos, pico_memoria_MB)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    resultado = func(*args, **kwargs)
    segundos = time.perf_counter() - t0
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return resultado, segundos, pico / 1024 ** 2

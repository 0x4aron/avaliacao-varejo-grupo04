# Medição de desempenho

- **Instrumentação:** `comum/medicao.py::medir`, que usa `time.perf_counter()` para o tempo e `tracemalloc` para o pico de memória.
- **Ambiente:** Python 3.12.3, NumPy 2.5.4, Linux.
- Os números abaixo foram copiados dos CSV gerados por `--modo completo`. Rodar de novo produz valores de tempo próximos (eles dependem da máquina) e resultados numéricos idênticos (fixados pela seed).

## Questão 1 — `questao_01/resultados/q1_escalabilidade.csv`

| Algoritmo | Entrada | Combinações | Tempo (s) | Pico mem. (MB) |
|---|---|---|---|---|
| Brute Force | k = 5 (pequena) | 6.144 | 0,106 | 1,15 |
| Brute Force | k = 6 | 18.432 | 0,322 | 3,81 |
| Brute Force | k = 7 (média) | 92.160 | 1,630 | 12,07 |
| Brute Force | k = 8 (grande) | 552.960 | 9,676 | 14,20 |
| GA (150 ger., P=120) | n = 10 | – | 0,742 | 0,09 |
| GA | n = 20 | – | 1,196 | 0,15 |
| GA | n = 30 | – | 1,673 | 0,22 |
| Greedy | n = 10 / 20 / 30 | – | 0,027 / 0,041 / 0,053 | 0,02 / 0,03 / 0,04 |

**Leitura dos resultados**
- O BF cresce exponencialmente (~6× por SKU adicional), como previsto em O((L+1)^k).
- O GA cresce linearmente em n.
- A memória do BF estabiliza a partir de k=7 porque as combinações são processadas em lotes de 50.000.

### Duas configurações do GA — `q1_comparacao_ga.csv`

| Config | P | Mutação | Torneio | Gerações até parar | Melhor fitness | Tempo (s) | Mem. (MB) |
|---|---|---|---|---|---|---|---|
| A | 120 | 0,05 | 3 | 120 | 331.414,63 | 1,55 | 0,22 |
| B | 240 | 0,02 | 4 | 261 | 331.414,63 | 3,16 | 0,43 |

**Leitura dos resultados**
- As duas configurações chegaram à mesma solução.
- A config B (população maior e mutação menor) explora mais devagar. Precisou do dobro de gerações e ~2× de tempo e memória, sem ganho.
- A config A é suficiente para n=30.
- A curva de convergência está em `q1_fitness_por_geracao.png`.

### Otimização demonstrada (antes/depois) — `q1_otimizacao_antes_depois.csv`

| Versão | Tempo p/ 2000 indivíduos | Pico mem. | Speedup |
|---|---|---|---|
| Antes: `avaliar_populacao_loop` (laço Python, recalcula demanda a cada avaliação) | 0,977 s | 0,03 MB | 1× |
| Depois: `avaliar_populacao` (tabela pré-computada + indexação NumPy) | 0,002 s | 0,55 MB | **~479×** |

**O que mudou**
- A demanda e o caixa de cada (SKU, opção) passaram a ser calculados uma única vez, em `construir_tabela`, ao custo de O(n·L) de memória.
- Avaliar um indivíduo virou uma soma de n consultas na tabela.
- A troca é um pouco mais de memória por muito menos tempo.
- O teste `test_fitness_vetorizado_igual_ao_laco` garante que as duas versões produzem o mesmo resultado.

## Questão 2 — `questao_02/resultados/q2_escalabilidade.csv`

| SKUs | Estoque | Σ estoque | Estados DP | MC constr. (s / MB) | DP+Greedy (s / MB) | Greedy (s) | MC valid. 5000 (s / MB) |
|---|---|---|---|---|---|---|---|
| 8 (pequena) | ×1 | 2.289 | 27.564 | 0,105 / 0,49 | 0,344 / 0,45 | 0,031 | 0,217 / 12,3 |
| 16 (média) | ×1 | 3.676 | 44.304 | 0,183 / 0,98 | 0,626 / 0,55 | 0,043 | 0,428 / 23,6 |
| 24 (grande) | ×1 | 4.573 | 55.164 | 0,273 / 1,17 | 0,826 / 0,56 | 0,052 | 0,625 / 34,9 |
| 24 | ×2 | 9.146 | 110.040 | 0,270 / 1,17 | 0,870 / 1,11 | 0,052 | 0,627 / 34,9 |
| 24 | ×4 | 18.292 | 219.792 | 0,270 / 1,17 | 0,957 / 2,19 | 0,052 | 0,628 / 34,9 |

**Leitura dos resultados**
- **Memória da DP:** cresce linearmente com o estoque (0,56 → 1,11 → 2,19 MB), confirmando O(T·S) por SKU.
- **Tempo da DP:** cresce pouco porque o laço em `s` é vetorizado.
- **Monte Carlo de validação:** domina a memória (O(C·T·n)) e cresce linearmente com n.
- **Monte Carlo de construção:** independe do estoque.

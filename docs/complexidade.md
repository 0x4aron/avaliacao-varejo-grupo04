# Complexidade assintótica (Big O)

## Variáveis

| Símbolo | Significado | Valor na instância G04 |
|---|---|---|
| `n` | nº de SKUs do catálogo | Q1: 30 · Q2: 24 |
| `L` | nº de níveis de preço (sem contar "suspender") | Q1: 7 · Q2: 6 |
| `k` | nº de SKUs do subconjunto do Brute Force | 7 |
| `P` | tamanho da população do GA | 120 (config A) / 240 (config B) |
| `G` | nº de gerações executadas pelo GA | ~120–260 (parada por paciência) |
| `D` | degraus de incentivo aplicados pelo Greedy da Q1 | ≤ n · 4 |
| `C` | nº de cenários Monte Carlo | 2000 (construção) / 5000 (validação) |
| `T` | nº de semanas | 12 |
| `S` | estoque inicial de um SKU (tamanho do eixo de estado) | até 576 |
| `K` | buckets (quantis) de demanda passados à DP | 7 |

---

## Questão 1

### Construção da tabela — `questao_01/modelo.py::construir_tabela`
- Tempo **O(n · L)**, memória **O(n · L)**: cada SKU × opção é calculado uma vez.
- Essa tabela é o que permite avaliar uma política em O(n) sem recalcular demanda.

### A) Brute Force — `questao_01/brute_force.py::brute_force`
- Tempo **O((L+1)^k · k)**. Há até (L+1)^k combinações e cada uma custa O(k) para avaliar.
- Na prática o expoente usa só as opções que respeitam a margem: Π|opções_i| = 92.160 para k=7.
- Trecho dominante: `valores = avaliar_populacao(tab, pop)` dentro do `while`, que avalia lotes de 50.000 combinações.
- Memória **O(B · k)**, com B = tamanho do lote (50.000). O `itertools.product` é preguiçoso, então as combinações nunca são todas materializadas.
- **Quando deixa de ser viável:** medido em `q1_escalabilidade.csv`, o tempo cresce ~6× a cada SKU a mais (k=7: 1,6 s; k=8: 9,7 s). Por extrapolação, k=10 levaria ~6 min e k=30 teria 8³⁰ ≈ 10²⁷ combinações. A alternativa é o **GA**, e o BF serve apenas de benchmark.

### B) Algoritmo Genético — `questao_01/genetic_algorithm.py::algoritmo_genetico`
- Tempo **O(G · P · n)**.
- Por geração:
  - seleção por torneio: O(P · torneio);
  - crossover e mutação: O(P · n);
  - avaliação: O(P · n).
- Trecho dominante: `fit = avaliar_populacao(tab, pop)`, uma indexação NumPy `tab.fitness[linhas, pop]` de tamanho P×n.
- Memória **O(P · n + G)**: a população atual, os filhos e o histórico de fitness. Populações antigas não são guardadas.
- **Viabilidade:** é linear em n. Com n=10, 20 e 30 o tempo foi 0,74 s, 1,20 s e 1,67 s para 150 gerações.
- **Limite:** o GA não garante o ótimo. Por isso é validado contra o BF (gap 0,00% no subconjunto).

### C) Greedy de incentivo — `questao_01/greedy.py::alocar_incentivo`
- Tempo **O(n + D · log n)**:
  - montar o heap custa O(n);
  - cada degrau aplicado faz um `heappop` e um `heappush`, O(log n) cada.
- Trecho dominante: o laço `while heap`.
- Memória **O(n)**: o heap guarda um candidato por SKU.
- **Viabilidade:** sempre viável, com 0,05 s para n=30.
- **Limite:** o greedy é ótimo só para benefícios marginais decrescentes por SKU, que é o caso da elasticidade constante. Num knapsack geral ele pode errar, e a alternativa seria DP sobre o orçamento.

## Questão 2

### A) Monte Carlo de construção — `questao_02/monte_carlo.py::estatisticas_construcao`
- Tempo **O(n · L · C log C)**:
  - para cada SKU e nível, a demanda é gerada em C cenários, O(C);
  - os quantis exigem ordenação, O(C log C).
- Trecho dominante: `np.quantile(d, probs)` dentro do laço duplo.
- Memória **O(C · n)** para os choques, mais **O(n · L · K)** para a tabela de saída.

### B) Programação Dinâmica — `questao_02/dynamic_programming.py::dp_sku`
- Tempo por SKU **O(T · S · L · K)**. São T·(S+1) estados e, para cada um, L decisões × K resultados de demanda.
- Total **O(T · L · K · Σ S_i)**.
- Trecho dominante: a linha `valor = peso * vendas * margem[j] + V[t + 1][s[:, None] - vendas]`. É ali que ocorre a **reutilização de subproblemas**, porque `V[t+1]` foi calculado uma vez e é consultado L·K vezes por estado.
- Na instância: 55.164 estados e 2.140.992 consultas, ou seja, cada subproblema é reutilizado ~39×.
- Memória **O(T · S)** por SKU (tabelas `V` e `politica`).
- **Viabilidade:** linear em S. Com estoque ×1, ×2 e ×4, a DP+Greedy levou 0,83 s, 0,87 s e 0,96 s.
- **Quando deixa de ser viável:** quando o estado passa a ser conjunto entre SKUs. Um estado com o estoque de todos os SKUs (Π S_i) ou com o caixa explodiria. A alternativa usada aqui é decompor por SKU e devolver o acoplamento (o caixa comum) ao Monte Carlo de validação.

### C) Greedy semanal — `questao_02/greedy.py::alocar_destaques`
- Tempo **O(T · n log n)**: uma ordenação por semana.
- Memória **O(T · n)** para a matriz de destaques.

### D) Monte Carlo de validação — `questao_02/monte_carlo.py::validar_plano`
- Tempo **O(C · T · n)**. O laço é em T×n e cada passo é vetorizado nos C cenários.
- Trecho dominante: `politicas[i][t, s]` (decisão adaptativa) e `np.minimum(s, round(d))`.
- Memória **O(C · T · n)** para os choques pré-sorteados (3 arrays C×T×n). Foram 34,9 MB com C=5000 e n=24.
- **Limite:** a memória cresce linearmente com C. Para C da ordem de 10⁶, a alternativa é sortear os choques semana a semana ou processar em lotes de cenários.

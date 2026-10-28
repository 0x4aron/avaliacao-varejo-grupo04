# Decisões de modelagem

## Geral
- **Código do grupo.** O código do grupo não foi informado pelo professor. Usei **G04**, meu número na chamada. Se o código oficial for outro, basta trocar `GRUPO` no Bloco 02 do notebook e reexecutar (ver README).
- **Dados de entrada.**
  - Os arquivos de `data/` são cópias byte a byte da saída do notebook oficial.
  - `comum/dados.py::verificar_integridade` recalcula o SHA-256 a cada execução e aborta se algum hash divergir do manifesto.
- **Cenários-surpresa.** São aplicados em memória por flags de linha de comando, sem editar nenhum arquivo de `data/`.

## Questão 1
1. **Demanda.**
   - A demanda tem elasticidade constante: `demanda(p) = demanda_base_90d · (p/preço_atual)^(-elasticidade)`.
   - As unidades vendidas são `min(estoque, demanda)`.
2. **Caixa.**
   - O caixa de cada SKU é `unidades · (p·(1−taxa) − custo − logística)`.
   - No preço atual, com margens negativas truncadas em 0, essa conta reproduz o `caixa_base_estimado` do gerador. A meta fica em 93% desse valor (teste `test_caixa_base_reproduz_gerador`).
3. **Preços permitidos.**
   - Cada SKU pode usar os níveis `0,80…1,10 × preço_atual` cuja margem percentual seja maior ou igual a `margem_minima_percentual`.
   - Cinco SKUs (Q1-003, Q1-010, Q1-022, Q1-023 e Q1-030) não têm **nenhum** preço viável: custo + logística + taxa superam até 1,10× o preço. Para eles existe a opção **"suspender venda"**, com 0 unidades. Ela está disponível para todos os SKUs.
4. **Restrições que acoplam os SKUs.** Sem elas o problema se separa por SKU e o Brute Force fica trivial.
   - **Limite de descontos:** no máximo `floor(30% · n)` SKUs com preço abaixo do atual. É o que evita uma "política de descontos indiscriminada".
   - **Giro mínimo:** o capital liberado em 90 dias (Σ unidades·custo) deve ser pelo menos o da política atual.
     - Motivo: subir todos os preços ao máximo aumenta o caixa por unidade, mas deixa estoque encalhado.
     - Implementação: a restrição é tratada por penalidade de R$5 por R$1 abaixo do mínimo.
5. **Custo de estoque parado.**
   - Penaliza o estoque que sobra: 25% a.a. sobre o custo, proporcional à idade do estoque.
   - Prioriza o giro de itens antigos.
6. **GA.**
   - **População e parada:** população de 120 indivíduos. O critério de parada é 300 gerações ou 60 gerações seguidas sem melhora.
   - **Seleção:** torneio de 3, simples e com pressão seletiva controlável.
   - **Crossover:** uniforme (90%). Os genes são independentes por SKU, então não há ordem a preservar.
   - **Mutação:** 5% por gene, re-sorteando apenas entre opções permitidas. Isso mantém a margem sempre válida.
   - **Elitismo:** os 4 melhores passam intactos, o que garante fitness monotônico.
   - **Semente da população:** um indivíduo representa a "política atual".
7. **Greedy de incentivo.**
   - **O que é o incentivo:** cupons em degraus de 2,5% (até 10%) sobre o preço do GA.
   - **Benefício:** ganho de caixa mais a redução do custo de estoque parado.
   - **Ordem de alocação:** sempre o degrau de maior benefício/R$ (heap).
   - **Como o orçamento é tratado:** como verba de marketing já provisionada.
   - **Resultado:** o relatório mostra o caixa **bruto** e o **líquido do incentivo**. Na instância, cada R$1 de incentivo devolve ~R$0,29 de caixa (R$25,1 mil de ganho para R$86,5 mil gastos). Logo, se a verba não estiver provisionada, a recomendação financeira é **não** gastá-la. O parâmetro `razao_minima=1.0` implementa essa regra.

## Questão 2
1. **Incerteza.** Há três fontes, todas log-normais com média 1 e o σ do config:
   - choque de demanda;
   - preço do concorrente, que entra na demanda com elasticidade cruzada de 0,8;
   - custo logístico.
2. **MC → DP.**
   - O Monte Carlo de construção (2000 cenários, seed+1) entrega à DP, para cada SKU e nível, os 7 quantis de demanda e a margem unitária média.
   - A DP **não** usa a fórmula analítica.
   - Níveis com margem média ≤ 0 ficam proibidos.
3. **DP.**
   - Estado (semana, estoque) por SKU, resolvida bottom-up.
   - O acoplamento via caixa entre SKUs não entra no estado, pois isso daria explosão combinatória. Ele é medido no MC de validação.
   - O parâmetro `--urgencia` desconta o caixa futuro para antecipar receita. Na instância não muda o resultado porque o risco já é nulo.
4. **Greedy.** Em cada semana há até 4 vagas de destaque. Cada destaque dá +20% de demanda e custa 4% da receita esperada.
   - A vaga é o recurso escasso que efetivamente limita: o orçamento semanal sobra. Por isso `--orcamento-mult 0.75` não muda o plano, e `--max-destaques` muda.
5. **Validação.**
   - São 5000 cenários **novos** (seed+2).
   - A decisão de preço é **adaptativa**: `politica[t, estoque_real]`.
   - Se o caixa não cobre a parcela na data, conta como falha e a dívida não é paga. Isso explica por que o "caixa final" de cenários com falha pode ser maior: a dívida ficou em aberto.

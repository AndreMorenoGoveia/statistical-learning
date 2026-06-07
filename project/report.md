# Previsão Probabilística de SSH com Redes de Quantis Implícitos e Codificação Temporal

**Autor:** André Moreno Goveia - 13682785
**Disciplina:** PCS5024 - Aprendizado Estatístico (2026)
**Trabalho:** EP de Séries Temporais - previsão probabilística da altura da superfície do mar (SSH) no Porto de Santos sob dados faltantes

---

## 1. Problema e Dados

A tarefa é prever a altura da superfície do mar (SSH) no Porto de Santos, uma série temporal
univariada amostrada a cada 10 minutos de 01/01/2020 a 30/06/2020 (25.613 observações de um único
canal ssh, em metros). O treino usa tudo antes de 01/06/2020 e o mês de junho é reservado para
teste.

O sinal é dominado pela maré. O recorte de 3 dias na Figura 1 mostra o característico regime
semidiurno misto do litoral brasileiro, com aproximadamente duas preamares e duas baixa-mares por
dia e uma pronunciada desigualdade diurna, modulado em horizontes mais longos pelo ciclo de
sizígia–quadratura e pela maré meteorológica.

![Visão geral dos dados](figures/data_overview.png)
*Figura 1 - Série completa de SSH com a divisão treino/teste (acima) e um recorte de 3 dias mostrando
a maré semidiurna dominante (abaixo).*

Este EP combina dois conceitos da literatura sobre a baseline de codificador–decodificador GRU
fornecida:

1. **Codificação temporal** no estilo de Vaswani et al. (2017)
2. uma cabeça de emissão por **Rede de Quantis Implícitos (IQN)** como em Gouttes et al. (2021), estudando seu comportamento à medida que uma fração controlada das observações é apagada para simular
dados faltantes.



---

## 2. Método

### 2.1 Backbone de previsão

O backbone é uma GRU sequência-a-sequência. Uma GRU **codificadora** consome a janela passada e a
resume em um estado oculto `h` e uma GRU **decodificadora** desenrola ao longo do horizonte de
previsão. Tanto a janela passada quanto a futura são construídas com uma janela deslizante e
preenchidas (padding); o padding é mascarado em toda perda e métrica. A SSH é padronizada com a
média/desvio-padrão do treino, e todos os erros reportados são calculados após desnormalizar de
volta para metros.

Outro fator da baseline fornecida é que o decodificador recebe uma entrada dummy de zeros em
cada passo futuro. Sem informação por passo, o decodificador só pode emitir uma continuação suave do
estado do codificador e não consegue reconstruir a fase da maré, que é a fraqueza que a
codificação temporal aborda.

### 2.2 Codificação temporal (Vaswani et al., 2017)

O tempo `t` de cada passo é mapeado para um vetor senoidal de dimensão `d` com frequências espaçadas
geometricamente:

```
TE(t)[2i]   = sin( t / P^(2i/d) )
TE(t)[2i+1] = cos( t / P^(2i/d) ),    P = 10000,  i = 0 … d/2 − 1
```

Esta é a codificação posicional do Transformer, mas aplicada a um tempo contínuo, de valor real
em vez de a um índice inteiro de token. Dois pontos de projeto são importantes aqui:

- O tempo é medido relativo à origem de previsão de cada janela (o primeiro passo a ser
  previsto), em minutos de modo que os passos passados carregam tempos negativos e os passos
  futuros, positivos.
- A amostragem irregular é tratada nativamente. Como a codificação é função do tempo (relativo)
  efetivo, lacunas na série não a distorcem: a rede sempre é informada de quão distante no tempo
  cada amostra está da origem de previsão, em vez de assumir uma cadência fixa de 10 minutos. Com
  `d = 16` os comprimentos de onda abrangem de minutos a alguns dias, cobrindo confortavelmente o
  contexto de 2 dias e o horizonte de 12 horas.

A codificação é usada em dois lugares:

- **Entrada do codificador:** as `d` features temporais são concatenadas à única feature de SSH,
  resultando em `d + 1` canais de entrada por passo passado.
- **Entrada do decodificador:** os zeros dummy são substituídos pela codificação temporal dos
  timestamps futuros, de modo que o decodificador é explicitamente condicionado a quando deve
  prever.

### 2.3 Cabeça de Rede de Quantis Implícitos (Gouttes et al., 2021)

Em vez de emitir um único ponto por passo, a cabeça IQN transforma o estado do decodificador `ψ_t`
em uma amostra da distribuição condicional ao reparametrizar um nível de quantil `τ ∼ U(0,1)`:

```
φ(τ) = ReLU( Σ_{i=0}^{n−1} cos(π i τ) w_i + b_i )      (embedding por cosseno, n = 64)
ŷ_t  = q( ψ_t ⊙ (1 + φ(τ)) )                           (⊙ = produto elemento a elemento)
```

onde `q` é um gerador feed-forward de duas camadas. Nenhuma família paramétrica é assumida para a
distribuição alvo. A cabeça é treinada minimizando a perda de quantil (pinball)

```
L_τ(y, ŷ) = max( τ (y − ŷ), (τ − 1)(y − ŷ) ),
```

que é o integrando do Continuous Ranked Probability Score (CRPS), calculando sua média sobre `τ`
recupera-se o CRPS a menos de um fator de dois. Durante o treino, um novo `τ` é amostrado para cada
passo de cada janela.

Na inferência o decodificador não é autorregressivo em SSH (ele lê apenas informação
temporal/posicional), de modo que a distribuição preditiva de cada passo é independente dado `ψ_t`.
Portanto obtemos os quantis preditivos diretamente consultando uma grade de valores de `τ`, em vez
de amostragem ancestral. A mediana (`τ = 0,5`) é usada como previsão pontual e pares simétricos
(p.ex. `τ = 0,05 / 0,95`) definem intervalos de previsão centrais.

### 2.4 Desenho experimental

Para isolar o efeito da codificação temporal, os dois modelos probabilísticos diferem
apenas em usar ou não a codificação:

- **IQN (sem cod. temporal)**: IQN-RNN baseline, entrada dummy de zeros no decodificador.
- **IQN + cod. temporal**: mesmo modelo com a codificação da Seção 2.2.

Cada um é treinado em três níveis de dados faltantes: **0 % (completo), 30%, 60%**, onde a
fração indicada de pontos é apagada uniformemente ao acaso de ambas as séries de treino e de
teste. Uma baseline pontual MSE determinística é treinada nos dados completos como referência para o valor agregado pela cabeça IQN.

Configurações comuns: janela passada de 2.880 min (≈2 dias), horizonte de 720 min (12 h), tamanho do
estado oculto da GRU 64, dimensão temporal `d = 16`, base de cossenos IQN `n = 64`, Adam a `1e-3`,
60 épocas. Todas as execuções compartilham a mesma semente, de modo que a inicialização dos pesos e
o embaralhamento são comparáveis.

**Métricas**: A acurácia pontual usa RMSE e MAE sobre a mediana (metros). A qualidade probabilística
usa CRPS (metros, pinball médio sobre uma grade de `τ`) e as perdas de quantil normalizadas de 50% / 90%
(QL50, QL90) de Gouttes et al. O teste de cobertura  compara a cobertura empírica
de cada intervalo central, a fração de alvos de teste que de fato caem dentro de `[q_{(1−c)/2},
q_{(1+c)/2}]` com nível nominal `c`. Um modelo bem calibrado se posiciona sobre a diagonal.

---

## 3. Resultados

Todos os números abaixo são sobre o conjunto de teste de junho, desnormalizados para metros. Cada
curva de perda converge suavemente com treino e teste acompanhando
de perto, de modo que os modelos não fiquem com overfitting.

**Tabela 1 - Métricas de teste.** RMSE/MAE/CRPS em metros, QL50/QL90 são perdas de quantil
normalizadas, Cov50/Cov90 são coberturas empíricas dos intervalos centrais de 50% / 90%.

| Configuração | Faltante | RMSE | MAE | CRPS | QL50 | QL90 | Cov50 | Cov90 |
|---|---|---|---|---|---|---|---|---|
| Baseline pontual MSE | 0% | 0.1294 | 0.1021 | - | - | - | - | - |
| IQN (sem cod. temporal) | 0% | 0.1224 | 0.0957 | 0.0709 | 0.1190 | 0.0549 | 0.469 | 0.892 |
| IQN + cod. temporal | 0% | **0.1211** | **0.0949** | **0.0702** | 0.1180 | 0.0533 | 0.475 | 0.912 |
| IQN (sem cod. temporal) | 30% | 0.1809 | 0.1384 | 0.1029 | 0.1719 | 0.0768 | 0.454 | 0.872 |
| IQN + cod. temporal | 30% | **0.1337** | **0.1035** | **0.0780** | 0.1286 | 0.0609 | 0.447 | 0.867 |
| IQN (sem cod. temporal) | 60% | 0.2303 | 0.1771 | 0.1308 | 0.2200 | 0.0972 | 0.448 | 0.875 |
| IQN + cod. temporal | 60% | **0.1422** | **0.1122** | **0.0866** | 0.1394 | 0.0610 | 0.401 | 0.764 |

### 3.1 IQN vs. baseline pontual (dados completos)

Em dados completos a cabeça IQN **iguala ou supera** a baseline MSE determinística na
acurácia pontual (RMSE da mediana 0,122 vs 0,129), entregando adicionalmente uma distribuição
preditiva completa sem custo extra em erro. Isto reproduz a afirmação central de Gouttes et al.: uma
cabeça de quantis não paramétrica não troca acurácia pontual por sua saída probabilística.

### 3.2 Efeito da codificação temporal sob dados faltantes (requisito 3)

Este é o resultado principal. Em dados completos os dois modelos IQN estão essencialmente
empatados (RMSE 0,121 vs 0,122). O quadro muda drasticamente à medida que dados são removidos:

![RMSE vs faltante](figures/rmse_vs_missing.png)
*Figura 2 - Acurácia pontual (RMSE) em função dos dados faltantes.*

![CRPS vs faltante](figures/crps_vs_missing.png)
*Figura 3 - Acurácia probabilística (CRPS) em função dos dados faltantes.*

A baseline (sem codificação) degrada acentuadamente - o RMSE sobe de 0,122 para 0,230 (+88 %) e
o CRPS de 0,071 para 0,131, porque, uma vez que amostras são descartadas, o k-ésimo passo do
decodificador não corresponde mais a um tempo fixo à frente, mas o modelo ainda o trata como se
correspondesse. O modelo com codificação temporal permanece quase plano com RMSE 0,121 para 0,142
(+17 %), CRPS 0,070 para 0,087, porque é informado do tempo verdadeiro (irregular) de cada passo.
A 60 % de dados faltantes a codificação reduz o RMSE em 38 % e o CRPS em 34 % em relação à
baseline. As perdas de quantil normalizadas QL50/QL90 contam a mesma história.

As Figuras 4 e 5 mostram a mesma janela de teste prevista a 60 % de faltantes: ambos os modelos
recuperam o formato da maré, mas o modelo com codificação temporal produz intervalos de previsão
visivelmente mais estreitos em torno da maré ascendente, ao passo que a baseline precisa se
proteger com uma banda mais larga.

![Previsão sem-TE 60%](figures/forecast_IQN_no_temporal_enc_60.png)
*Figura 4 - IQN baseline (sem codificação) a 60 % de faltantes.*

![Previsão TE 60%](figures/forecast_IQN_+_temporal_enc_60.png)
*Figura 5 - IQN + codificação temporal a 60 % de faltantes - intervalos mais estreitos.*

### 3.3 Teste de cobertura

O teste de cobertura verifica se os intervalos centrais estão calibrados: um intervalo de nível
nominal `c` deve conter o alvo uma fração `c` das vezes.

![Cobertura vs faltante](figures/coverage90_vs_missing.png)
*Figura 6 - Cobertura empírica do intervalo nominal de 90 % vs dados faltantes.*

![Confiabilidade da cobertura](figures/coverage_diagram.png)
*Figura 7 - Diagrama de confiabilidade a 60 % de faltantes (mais próximo da diagonal é melhor).*

Em dados completos ambos os modelos estão bem calibrados no nível de 90% (0,89 e 0,91, este último
essencialmente nominal). Uma sub-cobertura branda e consistente do intervalo central de 50 % é
visível em todo modelo (≈0,40–0,47 vs 0,50): a arquitetura IQN não impõe quantis monótonos em `τ`, e
os quantis internos acabam um pouco estreitos demais.

O efeito interessante está na alta taxa de faltantes. Os intervalos do modelo com codificação
permanecem estreitos mas não se alargam o suficiente - sua cobertura de 90% cai para 0,76
(superconfiante), enquanto a baseline mantém ≈0,87 simplesmente porque seus intervalos são largos e
não informativos (o preço de suas previsões pontuais ruins). Este é um trade-off entre
nitidez e calibração: a baseline compra uma cobertura de aparência nominal com bandas
inutilmente largas, ao passo que o modelo com codificação é mais nítido e muito mais acurado, porém
um pouco superconfiante em faltantes extremos. O CRPS, que recompensa nitidez e calibração
conjuntamente, favorece o modelo com codificação em todo nível de faltantes, fazendo com que ele seja o
melhor previsor probabilístico no geral.

---

## 4. Desafios

**Janelas irregulares e preenchidas.**: Ao remover pontos, cada janela passou a
ter um comprimento diferente, e meus primeiros resultados vinham contaminados sem que eu entendesse
por quê. Percebi que o padding estava vazando para a perda e inflando as métricas. Resolvi empacotando todas as sequências (`pack_padded_sequence`) e a calcular
toda perda e métrica sob uma máscara explícita de comprimento, de modo que o padding não
contaminasse o treino ou a avaliação.

**Orçamento computacional.**: Para rodar a grade completa de 7 modelos em uma única GPU de laptop, a
configuração reportada usa um contexto de 2 dias, horizonte de 12 horas e 60 épocas. O código expõe
todos esses parâmetros como argumentos de linha de comando (`--exp_*`), e uma única execução mais
longa pode ser lançada com `--mode single`.

---

## 5. Conclusão

Três achados se destacam:

1. A cabeça IQN iguala a acurácia pontual da baseline determinística enquanto produz uma
   distribuição preditiva calibrada - quantificação de incerteza essencialmente de graça.
2. A codificação temporal torna o previsor robusto a dados faltantes. Com dados completos ela é
   neutra, mas sua vantagem cresce monotonicamente com os faltantes, reduzindo o RMSE em 38 % e o
   CRPS em 34 % a 60 % de faltantes; sua acurácia degrada apenas 17 % em toda a faixa, contra 88 % da
   baseline.
3. O teste de cobertura expõe um trade-off entre nitidez e calibração: o modelo com codificação
   é mais nítido e mais acurado em toda parte (CRPS menor) mas levemente superconfiante em faltantes
   extremos, enquanto os intervalos mais largos da baseline parecem melhor calibrados apenas porque
   suas previsões pontuais são piores.

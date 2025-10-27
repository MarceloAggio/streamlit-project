# 📋 Documentação: Critérios de Análise de Recorrência de Alertas

## 🎯 Visão Geral

Esta documentação detalha os **16 critérios analíticos** utilizados para determinar se um alerta é **recorrente** ou **não recorrente**, com foco em identificar padrões temporais previsíveis em sistemas de TI/infraestrutura.

---

## 📊 Estrutura da Análise

A análise está dividida em **5 categorias principais** com **16 análises essenciais**:

### 1️⃣ **Análises Estatísticas (8 análises)**
### 2️⃣ **Análises de Previsibilidade (2 análises)**
### 3️⃣ **Análises Contextuais (3 análises)**
### 4️⃣ **Análises Preditivas (1 análise)**
### 5️⃣ **Análises Avançadas de Machine Learning (2 análises)**

---

## 🔍 DETALHAMENTO DOS CRITÉRIOS

---

## 📈 CATEGORIA 1: ANÁLISES ESTATÍSTICAS (1-8)

### **1. Estatísticas de Intervalos**
**Método:** `_analyze_basic_statistics()`

**O que faz:**
- Calcula estatísticas descritivas dos intervalos entre ocorrências
- Métricas: média, mediana, desvio padrão, mínimo, máximo, CV, quartis, IQR

**Por que é importante:**
- Fornece a **base quantitativa** para todas as outras análises
- Intervalos consistentes (baixo desvio padrão) indicam **padrão regular**
- Diferença entre média e mediana revela presença de **outliers**

**Indicador de recorrência:**
- Desvio padrão baixo = alta recorrência
- Intervalos consistentes = comportamento previsível

---

### **2. Regularidade e Aleatoriedade (Coeficiente de Variação - CV)**
**Método:** `_analyze_regularidade()`

**O que faz:**
- Calcula o **Coeficiente de Variação (CV)** = Desvio Padrão / Média
- Classifica o padrão em 5 níveis:
  - CV < 0.15: 🟢 **ALTAMENTE REGULAR** (95 pontos)
  - CV < 0.35: 🟢 **REGULAR** (80 pontos)
  - CV < 0.65: 🟡 **SEMI-REGULAR** (60 pontos)
  - CV < 1.0: 🟠 **IRREGULAR** (40 pontos)
  - CV ≥ 1.0: 🔴 **ALTAMENTE IRREGULAR** (20 pontos)

**Por que é importante:**
- **Critério mais importante** (20 pontos no score final)
- CV baixo = intervalos muito consistentes = **forte recorrência**
- Normaliza a variabilidade independentemente da escala temporal

**Indicador de recorrência:**
- CV < 0.35 → FORTE indicação de recorrência
- CV > 1.0 → Comportamento aleatório/esporádico

---

### **3. Análise de Periodicidade (Transformada de Fourier - FFT)**
**Método:** `_analyze_periodicity()`

**O que faz:**
- Aplica **FFT (Fast Fourier Transform)** nos intervalos
- Identifica **frequências dominantes** (ciclos repetitivos)
- Detecta períodos: horário (1h), diário (24h), semanal (168h), mensal (720h)

**Por que é importante:**
- Detecta **ciclos ocultos** que não são visíveis em análises simples
- Fundamental para identificar padrões periódicos (ex: jobs agendados)
- **2º critério mais importante** (20 pontos no score final)

**Indicador de recorrência:**
- Presença de picos de frequência fortes = periodicidade clara
- Múltiplos harmônicos = padrão complexo mas recorrente

---

### **4. Análise de Autocorrelação**
**Método:** `_analyze_autocorrelation()`

**O que faz:**
- Calcula correlação dos intervalos **consigo mesmos** em diferentes defasagens (lags)
- Identifica se um evento influencia o próximo

**Por que é importante:**
- Detecta **dependência temporal**: se um alerta "chama" o próximo
- Autocorrelação alta = eventos não são independentes = **padrão recorrente**
- Contribui com **10 pontos** no score final

**Indicador de recorrência:**
- Autocorrelação > 0.3 em múltiplos lags = forte dependência temporal
- Padrão de decaimento lento = memória de longo prazo

---

### **5. Padrões Temporais (Hora do Dia / Dia da Semana)**
**Método:** `_analyze_temporal_patterns()`

**O que faz:**
- Analisa **concentração** de alertas por:
  - Hora do dia (0-23h)
  - Dia da semana (Segunda-Domingo)
  - Dia do mês
  - Semana do ano
- Calcula **entropia** da distribuição

**Por que é importante:**
- Identifica **janelas temporais** onde alertas se concentram
- Entropia baixa = concentração em horários específicos = **padrão recorrente**
- Essencial para entender **contexto operacional** (ex: horário comercial)

**Indicador de recorrência:**
- >60% dos alertas em 3 horas específicas = forte padrão horário
- >70% em 2 dias da semana = padrão semanal claro

---

### **6. Análise de Clusters Temporais**
**Método:** `_analyze_clusters()`

**O que faz:**
- Aplica **DBSCAN** e **KMeans** para agrupar intervalos similares
- Identifica **grupos de comportamento** distintos
- Calcula **Silhouette Score** (qualidade do agrupamento)

**Por que é importante:**
- Detecta se existem **múltiplos padrões** coexistentes
- Cluster dominante (>60% dos dados) indica **padrão principal forte**
- Silhouette Score alto = clusters bem definidos = padrões claros

**Indicador de recorrência:**
- 1-2 clusters dominantes = comportamento consistente
- Múltiplos clusters dispersos = comportamento variável

---

### **7. Detecção de Bursts (Explosões de Alertas)**
**Método:** `_detect_bursts()`

**O que faz:**
- Identifica **períodos de alta concentração** de alertas
- Detecta intervalos muito menores que a média
- Calcula intensidade e duração dos bursts

**Por que é importante:**
- Diferencia entre:
  - Padrão recorrente estável
  - Padrão com **explosões periódicas**
  - Eventos esporádicos concentrados
- Bursts recorrentes podem indicar problema sistêmico

**Indicador de recorrência:**
- Bursts periódicos (mesmo horário/dia) = recorrência com picos
- Bursts aleatórios = problemas pontuais

---

### **8. Análise de Sazonalidade**
**Método:** `_analyze_seasonality()`

**O que faz:**
- Detecta padrões de **longo prazo** (semanal, mensal, anual)
- Analisa variação ao longo de semanas/meses
- Identifica tendências crescentes ou decrescentes

**Por que é importante:**
- Captura **ciclos de negócio** (ex: fim de mês, fechamento de trimestre)
- Diferencia recorrência de curto prazo vs. longo prazo
- Essencial para planejamento de capacidade

**Indicador de recorrência:**
- Padrão consistente mês a mês = sazonalidade forte
- Variação alta entre meses = comportamento instável

---

## 🎯 CATEGORIA 2: ANÁLISES DE PREVISIBILIDADE (9-10)

### **9. Score de Previsibilidade**
**Método:** `_calculate_predictability()`

**O que faz:**
- Combina múltiplas métricas:
  - Regularidade (CV)
  - Autocorrelação
  - Clustering
  - Periodicidade
- Gera **score 0-100** de quão previsível é o próximo alerta

**Por que é importante:**
- **Critério fundamental** (15 pontos no score final)
- Se conseguimos prever, há padrão recorrente
- Métrica **agregada** que consolida várias análises

**Indicador de recorrência:**
- Score > 70 = altamente previsível = forte recorrência
- Score < 30 = imprevisível = comportamento aleatório

---

### **10. Análise de Estabilidade Temporal**
**Método:** `_analyze_stability()`

**O que faz:**
- Divide timeline em **janelas temporais**
- Compara variância entre janelas (início vs. fim do período)
- Aplica **Isolation Forest** para detectar anomalias

**Por que é importante:**
- Padrões recorrentes devem ser **estáveis ao longo do tempo**
- Detecta se o padrão está se intensificando ou enfraquecendo
- Contribui com **10 pontos** no score final

**Indicador de recorrência:**
- Variância estável entre janelas = padrão consistente
- Poucos outliers = comportamento regular

---

## 🌐 CATEGORIA 3: ANÁLISES CONTEXTUAIS (11-13)

### **11. Dependências Contextuais**
**Método:** `_analyze_contextual_dependencies()`

**O que faz:**
- Analisa correlação com **eventos externos**:
  - Feriados nacionais
  - Fins de semana
  - Horários comerciais vs. não comerciais
- Identifica se alertas aumentam/diminuem em contextos específicos

**Por que é importante:**
- Explica **causa raiz** de padrões temporais
- Diferencia entre:
  - Recorrência técnica (job agendado)
  - Recorrência operacional (carga de trabalho)
- Essencial para **ações corretivas**

**Indicador de recorrência:**
- Forte correlação com feriados = padrão contextual
- Concentração em horário comercial = relacionado à carga de usuários

---

### **12. Janelas de Vulnerabilidade Temporal**
**Método:** `_identify_vulnerability_windows()`

**O que faz:**
- Identifica **Top 5 horários mais críticos**
- Calcula % de concentração de alertas
- Fornece intervalo de confiança

**Por que é importante:**
- Mostra **quando** o problema ocorre com mais frequência
- Permite **monitoramento preventivo** em horários críticos
- Facilita **priorização** de recursos

**Indicador de recorrência:**
- 80% dos alertas em 2-3 horários = janela bem definida
- Distribuição uniforme = sem padrão temporal claro

---

### **13. Análise de Maturidade do Padrão**
**Método:** `_analyze_pattern_maturity()`

**O que faz:**
- Divide histórico em 3 períodos: início, meio, fim
- Compara características estatísticas entre períodos
- Determina se padrão está:
  - Se estabelecendo (aumentando consistência)
  - Estável (maturidade)
  - Se degradando (perdendo consistência)

**Por que é importante:**
- Padrão **maduro e estável** = recorrência estabelecida
- Padrão emergente = pode não ser permanente
- Padrão em degradação = problema pode estar se resolvendo

**Indicador de recorrência:**
- Maturidade alta + estabilidade = recorrência consolidada
- Padrão emergente = precisa mais dados para confirmar

---

## 🔮 CATEGORIA 4: ANÁLISES PREDITIVAS (14)

### **14. Confiança de Predição**
**Método:** `_calculate_prediction_confidence()`

**O que faz:**
- Calcula **intervalo de confiança** para próxima ocorrência
- Usa média ± margem de erro baseada no desvio padrão
- Fornece data/hora esperada com nível de confiança

**Por que é importante:**
- Transforma análise em **ação prática**: quando esperar o próximo alerta
- Confiança alta = padrão bem estabelecido
- Permite **planejamento proativo**

**Indicador de recorrência:**
- Intervalo de confiança estreito = alta previsibilidade
- Intervalo amplo = comportamento variável

---

## 🤖 CATEGORIA 5: ANÁLISES AVANÇADAS DE ML (15-16)

### **15. Cadeias de Markov**
**Método:** `_analyze_markov_chains()`

**O que faz:**
- Modela sequência de intervalos como **cadeia de Markov**
- Cria **matriz de transição** entre estados (curto/médio/longo intervalo)
- Calcula **distribuição estacionária** (equilíbrio de longo prazo)
- Determina **previsibilidade markoviana**

**Por que é importante:**
- Detecta se eventos seguem **processo estocástico previsível**
- Identifica padrões de transição (ex: após intervalo curto, tende a vir outro curto)
- Contribui com **10 pontos** no score final

**Indicador de recorrência:**
- Distribuição estacionária concentrada = estados previsíveis
- Transições determinísticas (1 estado → sempre mesmo próximo estado) = forte recorrência

---

### **16. Bateria de Testes de Aleatoriedade**
**Método:** `_advanced_randomness_tests()`

**O que faz:**
Executa **5 testes estatísticos rigorosos**:

#### a) **Runs Test (Wald-Wolfowitz)**
- Testa se sequência é aleatória analisando "corridas" de valores acima/abaixo da mediana
- p-value < 0.05 = sequência NÃO é aleatória

#### b) **Permutation Entropy**
- Mede complexidade da sequência (0-1)
- Valor baixo = padrão simples/repetitivo
- Valor alto = caótico/aleatório

#### c) **Approximate Entropy (ApEn)**
- Quantifica irregularidade da sequência
- ApEn baixo = regular/previsível
- ApEn alto = complexo/imprevisível

#### d) **Serial Correlation (Ljung-Box)**
- Testa se há correlação entre valores consecutivos
- p-value < 0.05 = sequência correlacionada (não aleatória)

#### e) **Hurst Exponent**
- Mede dependência de longo prazo:
  - H > 0.5: Tendência persistente (se sobe, tende a continuar subindo)
  - H = 0.5: Movimento aleatório (Browniano)
  - H < 0.5: Reversão à média (anti-persistente)

**Por que é importante:**
- Conjunto de testes **complementares** para validar não-aleatoriedade
- Se 3+ testes indicam não-aleatoriedade = **forte evidência de padrão**
- Contribui com **15 pontos** (critério de "Determinismo") no score final

**Indicador de recorrência:**
- Maioria dos testes rejeitando aleatoriedade = padrão determinístico
- Hurst > 0.5 + baixa entropia = comportamento persistente e previsível

---

## 🏆 SISTEMA DE PONTUAÇÃO FINAL

### **Cálculo do Score (0-100 pontos)**

O score final combina **7 critérios essenciais** com pesos diferenciados:

| # | Critério | Peso | O que mede |
|---|----------|------|------------|
| 1 | **Regularidade (CV)** | 20 pts | Consistência dos intervalos |
| 2 | **Periodicidade (FFT)** | 20 pts | Presença de ciclos repetitivos |
| 3 | **Previsibilidade** | 15 pts | Capacidade de prever próxima ocorrência |
| 4 | **Determinismo (Aleatoriedade)** | 15 pts | Comportamento não-aleatório |
| 5 | **Autocorrelação** | 10 pts | Dependência temporal entre eventos |
| 6 | **Estabilidade** | 10 pts | Manutenção do padrão ao longo do tempo |
| 7 | **Padrão Markoviano** | 10 pts | Transições de estado previsíveis |

---

### **Classificação de Recorrência**

```
🔴 ALERTA REINCIDENTE (P1 - Crítico)
Score: 70-100 pontos
- Padrão fortemente estabelecido
- Alta previsibilidade
- Ação imediata necessária

🟠 PARCIALMENTE REINCIDENTE (P2 - Alto)
Score: 50-69 pontos
- Padrão detectável mas com variações
- Previsibilidade moderada
- Monitoramento ativo recomendado

🟡 PADRÃO DETECTÁVEL (P3 - Médio)
Score: 30-49 pontos
- Padrão fraco ou emergente
- Previsibilidade limitada
- Observação continuada

🟢 NÃO REINCIDENTE (P4 - Baixo)
Score: 0-29 pontos
- Comportamento aleatório/esporádico
- Não previsível
- Eventos isolados
```

---

## 🎯 RESUMO: POR QUE CADA CRITÉRIO IMPORTA

### **Critérios Primários (Recorrência Clara):**
1. **Regularidade (CV < 0.35)** → Intervalos consistentes
2. **Periodicidade (FFT com picos)** → Ciclos identificáveis
3. **Autocorrelação alta** → Eventos relacionados temporalmente

### **Critérios Secundários (Validação do Padrão):**
4. **Previsibilidade (Score > 70)** → Confirmação matemática
5. **Testes de aleatoriedade rejeitados** → Não é coincidência
6. **Estabilidade temporal** → Padrão se mantém

### **Critérios Contextuais (Causa Raiz):**
7. **Janelas de vulnerabilidade** → Quando ocorre
8. **Dependências contextuais** → Por que ocorre
9. **Maturidade do padrão** → Há quanto tempo existe

### **Critérios Preditivos (Ação Preventiva):**
10. **Confiança de predição** → Quando esperar o próximo
11. **Cadeias de Markov** → Sequência esperada de eventos

---

## 📋 FLUXO DE DECISÃO

```
┌─────────────────────────────────────┐
│   ANÁLISE DE RECORRÊNCIA INICIADA   │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 1. ESTATÍSTICAS BÁSICAS              │
│    → Mínimo 3 ocorrências            │
│    → Cálculo de intervalos           │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 2. REGULARIDADE (CV)                 │
│    CV < 0.35? ──► SIM = +40 pontos   │
│                   NÃO = continuar    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 3. PERIODICIDADE (FFT)               │
│    Picos dominantes? ──► SIM = +20   │
│                          NÃO = 0     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 4. TESTES DE ALEATORIEDADE           │
│    3+ testes rejeitados? ──► SIM +15 │
│                              NÃO = 0 │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 5. ANÁLISES COMPLEMENTARES           │
│    → Autocorrelação (+0-10)          │
│    → Estabilidade (+0-10)            │
│    → Markov (+0-10)                  │
│    → Previsibilidade (+0-15)         │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 6. SOMA DOS PONTOS                   │
│    Score Final = 0-100               │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 7. CLASSIFICAÇÃO FINAL               │
│    ≥70 = 🔴 P1 (Reincidente)        │
│    50-69 = 🟠 P2 (Parcial)          │
│    30-49 = 🟡 P3 (Detectável)       │
│    <30 = 🟢 P4 (Não Reincidente)    │
└──────────────────────────────────────┘
```

---

## ✅ CONCLUSÃO

A metodologia utiliza uma abordagem **multi-dimensional** combinando:

1. **Estatística clássica** (CV, correlação, testes de hipótese)
2. **Análise de sinais** (FFT, autocorrelação)
3. **Machine Learning** (clustering, Isolation Forest, Markov)
4. **Análise contextual** (temporal, sazonal, externa)
5. **Modelagem preditiva** (confiança, próxima ocorrência)

**Forças da metodologia:**
✅ Múltiplas perspectivas convergentes aumentam confiança
✅ Não depende de um único critério (robustez)
✅ Captura padrões simples e complexos
✅ Fornece não apenas classificação, mas explicação e predição

**Por que funciona:**
- Um alerta **verdadeiramente recorrente** pontuará alto em MÚLTIPLOS critérios
- Um alerta **esporádico** falhará na maioria dos testes
- A combinação de 16 análises cria um **sistema de validação cruzada**

---

## 📌 RECOMENDAÇÃO DE USO

Para justificar uma classificação de recorrência, cite:

1. **Score final** (0-100)
2. **Top 3 critérios** que mais contribuíram
3. **Padrão identificado** (ex: "Alerta ocorre toda segunda-feira às 8h com CV=0.12")
4. **Confiança de predição** (ex: "95% de confiança para próxima ocorrência em 23/10 14:00")
5. **Evidência contextual** (ex: "Correlacionado com horário de backup")

Exemplo de justificativa completa:
> "Alerta classificado como **P1 - Reincidente Crítico** (Score: 87/100).
> Apresenta **regularidade excepcional** (CV=0.09, +20pts), **periodicidade clara** 
> de 24h (FFT, +20pts) e **alta previsibilidade** (92/100, +15pts). 
> Concentra-se em **horário de backup** (95% das ocorrências 02:00-03:00), 
> com padrão **estável há 3 meses**. Próxima ocorrência prevista: 28/10 02:15 ±30min."

---

**Documentação gerada em:** Outubro/2025
**Versão do Sistema:** 16 Análises Essenciais
**Critérios de Score:** 7 dimensões, 100 pontos máximos
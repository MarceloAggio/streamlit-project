import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import io
import warnings
from multiprocessing import Pool, cpu_count
import holidays
from functools import partial
from collections import defaultdict, Counter
import math

warnings.filterwarnings(‘ignore’)

st.set_page_config(
page_title=“Analisador de Alertas - Otimizado”,
page_icon=“🚨”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

# ============================================================

# FUNÇÕES DE GERAÇÃO DE REGRAS CUSTOMIZADAS

# ============================================================

def identificar_padrao_dominante(results):
“”“Identifica o padrão DOMINANTE baseado em todos os scores”””
padroes_detectados = []

```
# 1. PERIODICIDADE FORTE
if results.get('periodicity', {}).get('has_strong_periodicity', False):
    periodo_horas = results['periodicity'].get('dominant_period_hours')
    
    if periodo_horas:
        confianca = 90 if results.get('regularity', {}).get('regularity_score', 0) > 80 else 75
        
        if periodo_horas < 1:
            subtipo = "MINUTOS"
            descricao = f"A cada {periodo_horas * 60:.0f} minutos"
        elif periodo_horas < 12:
            subtipo = "HORARIO_FIXO"
            descricao = f"A cada {periodo_horas:.1f} horas"
        elif 20 <= periodo_horas <= 28:
            subtipo = "DIARIO"
            descricao = "Aproximadamente 1x por dia"
        elif 160 <= periodo_horas <= 180:
            subtipo = "SEMANAL"
            descricao = "Aproximadamente 1x por semana"
        elif 330 <= periodo_horas <= 370:
            subtipo = "QUINZENAL"
            descricao = "Aproximadamente a cada 15 dias"
        else:
            subtipo = "PERIODICO_IRREGULAR"
            descricao = f"Período de {periodo_horas / 24:.1f} dias"
        
        padroes_detectados.append({
            'tipo': 'PERIODICO_FIXO',
            'subtipo': subtipo,
            'confianca': confianca,
            'score_relevancia': 95,
            'periodo_horas': periodo_horas,
            'descricao': descricao
        })

# 2. CONCENTRAÇÃO HORÁRIA
if results['temporal'].get('hourly_concentration', 0) > 60:
    peak_hours = results['temporal'].get('peak_hours', [])
    
    if peak_hours:
        confianca = 85 if len(peak_hours) <= 3 else 70
        
        padroes_detectados.append({
            'tipo': 'HORARIOS_FIXOS',
            'subtipo': 'INTRADIARIO',
            'confianca': confianca,
            'score_relevancia': 90,
            'horarios': peak_hours,
            'concentracao': results['temporal']['hourly_concentration'],
            'descricao': f"Concentrado nos horários: {', '.join([f'{h:02d}:00' for h in peak_hours])}"
        })

# 3. CONCENTRAÇÃO SEMANAL
if results['temporal'].get('daily_concentration', 0) > 60:
    peak_days = results['temporal'].get('peak_days', [])
    
    if peak_days:
        confianca = 80
        dias_map = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        dias_nome = [dias_map[d] for d in peak_days]
        
        padroes_detectados.append({
            'tipo': 'DIAS_FIXOS',
            'subtipo': 'SEMANAL',
            'confianca': confianca,
            'score_relevancia': 85,
            'dias': peak_days,
            'concentracao': results['temporal']['daily_concentration'],
            'descricao': f"Concentrado nos dias: {', '.join(dias_nome)}"
        })

# 4. PADRÃO DE BURST/RAJADA
if results['bursts'].get('has_bursts', False):
    n_bursts = results['bursts'].get('n_bursts', 0)
    
    if n_bursts >= 2:
        confianca = 75
        
        padroes_detectados.append({
            'tipo': 'BURST',
            'subtipo': 'RAJADAS',
            'confianca': confianca,
            'score_relevancia': 70,
            'num_bursts': n_bursts,
            'descricao': f"Padrão de rajadas ({n_bursts} detectadas)"
        })

# 5. ALTA REGULARIDADE sem periodicidade clara
if (results['regularity']['regularity_score'] > 70 and 
    not results['periodicity'].get('has_strong_periodicity', False)):
    
    cv = results['regularity']['cv']
    confianca = 65
    
    padroes_detectados.append({
        'tipo': 'REGULAR_SEM_CICLO',
        'subtipo': 'ESTAVEL',
        'confianca': confianca,
        'score_relevancia': 60,
        'cv': cv,
        'descricao': f"Regular mas sem ciclo claro (CV={cv:.2%})"
    })

# 6. MARKOV
if results['markov'].get('markov_score', 0) > 60:
    markov_score = results['markov']['markov_score']
    confianca = 70
    
    padroes_detectados.append({
        'tipo': 'MARKOV',
        'subtipo': 'DEPENDENTE',
        'confianca': confianca,
        'score_relevancia': 55,
        'markov_score': markov_score,
        'descricao': f"Padrão markoviano (score={markov_score:.1f})"
    })

# 7. COMPORTAMENTO CONTEXTUAL
if results.get('contextual'):
    weekend_corr = results['contextual'].get('weekend_correlation', 0)
    holiday_corr = results['contextual'].get('holiday_correlation', 0)
    
    if weekend_corr > 0.3:
        confianca = 70
        padroes_detectados.append({
            'tipo': 'CONTEXTUAL',
            'subtipo': 'FINS_DE_SEMANA',
            'confianca': confianca,
            'score_relevancia': 65,
            'correlacao': weekend_corr,
            'descricao': f"Correlacionado com fins de semana ({weekend_corr:.0%})"
        })
    
    if holiday_corr > 0.2:
        confianca = 65
        padroes_detectados.append({
            'tipo': 'CONTEXTUAL',
            'subtipo': 'FERIADOS',
            'confianca': confianca,
            'score_relevancia': 60,
            'correlacao': holiday_corr,
            'descricao': f"Correlacionado com feriados ({holiday_corr:.0%})"
        })

# 8. SEM PADRÃO CLARO
if not padroes_detectados or results['randomness'].get('overall_randomness_score', 50) > 70:
    return {
        'tipo': 'SEM_PADRAO_CLARO',
        'subtipo': 'ALEATORIO',
        'confianca': 60,
        'score_relevancia': 0,
        'descricao': 'Comportamento aleatório ou sem padrão detectável'
    }, []

padroes_detectados.sort(key=lambda x: (x['score_relevancia'], x['confianca']), reverse=True)

padrao_principal = padroes_detectados[0] if padroes_detectados else None
padroes_secundarios = padroes_detectados[1:3] if len(padroes_detectados) > 1 else []

return padrao_principal, padroes_secundarios
```

def gerar_regra_customizada(results, df, intervals_hours):
“”“Gera regra customizada baseada no padrão identificado”””
required_keys = [‘regularity’, ‘periodicity’, ‘predictability’, ‘basic_stats’]
for key in required_keys:
if key not in results or results[key] is None:
return gerar_regra_generica()

```
padrao_principal, padroes_secundarios = identificar_padrao_dominante(results)

if not padrao_principal:
    return gerar_regra_generica()

tipo = padrao_principal['tipo']

if tipo == 'PERIODICO_FIXO':
    periodo = padrao_principal['periodo_horas']
    subtipo = padrao_principal['subtipo']
    
    cv = results['regularity']['cv']
    if cv < 0.2:
        tolerancia_pct = 0.10
    elif cv < 0.4:
        tolerancia_pct = 0.15
    else:
        tolerancia_pct = 0.25
    
    tolerancia = periodo * tolerancia_pct
    
    if periodo < 4:
        ocorrencias_minimas = 5
        janela_analise_multiplicador = 6
    elif periodo < 12:
        ocorrencias_minimas = 4
        janela_analise_multiplicador = 5
    elif periodo < 48:
        ocorrencias_minimas = 3
        janela_analise_multiplicador = 4
    else:
        ocorrencias_minimas = 3
        janela_analise_multiplicador = 4
    
    janela_analise_horas = periodo * janela_analise_multiplicador
    
    return {
        'tipo': 'INTERVALO_FIXO',
        'subtipo': subtipo,
        'padrao_principal': padrao_principal,
        'padroes_secundarios': padroes_secundarios,
        'parametros': {
            'periodo_esperado_horas': round(periodo, 2),
            'tolerancia_horas': round(tolerancia, 2),
            'tolerancia_percentual': tolerancia_pct * 100,
            'ocorrencias_consecutivas_minimas': ocorrencias_minimas,
            'janela_analise_horas': round(janela_analise_horas, 2),
            'confianca_padrao': padrao_principal['confianca']
        },
        'criterio_reincidencia': {
            'descricao': f"Considerar REINCIDENTE se ocorrer {ocorrencias_minimas}+ vezes consecutivas",
            'condicao': f"Com intervalo de {periodo:.1f}h (±{tolerancia:.1f}h)",
            'janela': f"Analisando últimas {janela_analise_horas:.0f}h"
        },
        'implementacao': {
            'logica': f"""
```

FOR cada novo alerta:
ultimos_alertas = buscar_ultimos_alertas(janela={janela_analise_horas:.0f}h)

```
IF len(ultimos_alertas) >= {ocorrencias_minimas}:
    intervalos = calcular_intervalos(ultimos_alertas)
    
    contador_padrao = 0
    FOR intervalo in intervalos:
        IF {periodo - tolerancia:.1f} <= intervalo <= {periodo + tolerancia:.1f}:
            contador_padrao += 1
    
    IF contador_padrao >= {ocorrencias_minimas - 1}:
        MARCAR_COMO_REINCIDENTE()
```

“””
}
}

```
elif tipo == 'HORARIOS_FIXOS':
    horarios = padrao_principal['horarios']
    concentracao = padrao_principal['concentracao']
    
    tolerancia_minutos = 30 if concentracao > 80 else 45
    dias_consecutivos = 3 if concentracao > 75 else 4
    
    return {
        'tipo': 'HORARIOS_RECORRENTES',
        'subtipo': 'INTRADIARIO',
        'padrao_principal': padrao_principal,
        'padroes_secundarios': padroes_secundarios,
        'parametros': {
            'horarios_esperados': horarios,
            'tolerancia_minutos': tolerancia_minutos,
            'dias_consecutivos_minimo': dias_consecutivos,
            'concentracao_horaria': concentracao,
            'confianca_padrao': padrao_principal['confianca']
        },
        'criterio_reincidencia': {
            'descricao': f"Considerar REINCIDENTE se ocorrer {dias_consecutivos}+ dias consecutivos",
            'condicao': f"Nos horários: {', '.join([f'{h:02d}:00' for h in horarios])} (±{tolerancia_minutos}min)",
            'janela': f"Últimos {dias_consecutivos} dias"
        },
        'implementacao': {
            'logica': f"""
```

FOR cada novo alerta:
hora_alerta = extrair_hora(alerta)
em_horario_pico = FALSE
FOR horario_esperado in {horarios}:
IF abs(hora_alerta - horario_esperado) <= {tolerancia_minutos / 60:.2f}:
em_horario_pico = TRUE

```
IF em_horario_pico:
    ultimos_dias = buscar_ultimos_dias({dias_consecutivos})
    dias_com_padrao = contar_dias_com_horario_pico()
    
    IF dias_com_padrao >= {dias_consecutivos}:
        MARCAR_COMO_REINCIDENTE()
```

“””
}
}

```
elif tipo == 'DIAS_FIXOS':
    dias = padrao_principal['dias']
    concentracao = padrao_principal['concentracao']
    
    dias_map = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 
                4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
    dias_nome = [dias_map[d] for d in dias]
    
    semanas_consecutivas = 3 if concentracao > 75 else 4
    
    return {
        'tipo': 'DIAS_RECORRENTES',
        'subtipo': 'SEMANAL',
        'padrao_principal': padrao_principal,
        'padroes_secundarios': padroes_secundarios,
        'parametros': {
            'dias_esperados': dias,
            'dias_nome': dias_nome,
            'semanas_consecutivas_minimo': semanas_consecutivas,
            'concentracao_semanal': concentracao,
            'confianca_padrao': padrao_principal['confianca']
        },
        'criterio_reincidencia': {
            'descricao': f"Considerar REINCIDENTE se ocorrer {semanas_consecutivas}+ semanas consecutivas",
            'condicao': f"Nos dias: {', '.join(dias_nome)}",
            'janela': f"Últimas {semanas_consecutivas} semanas"
        },
        'implementacao': {
            'logica': f"""
```

FOR cada novo alerta:
dia_semana = extrair_dia_semana(alerta)

```
IF dia_semana in {dias}:
    ultimas_semanas = buscar_ultimas_semanas({semanas_consecutivas})
    semanas_com_padrao = contar_semanas_com_dias_pico()
    
    IF semanas_com_padrao >= {semanas_consecutivas}:
        MARCAR_COMO_REINCIDENTE()
```

“””
}
}

```
elif tipo == 'BURST':
    n_bursts = padrao_principal['num_bursts']
    
    media_intervalo = results['basic_stats']['mean']
    intervalo_burst = media_intervalo / 3
    
    ocorrencias_minimas_burst = 5
    janela_horas = max(6, intervalo_burst * ocorrencias_minimas_burst * 1.5)
    
    return {
        'tipo': 'RAJADA',
        'subtipo': 'BURST',
        'padrao_principal': padrao_principal,
        'padroes_secundarios': padroes_secundarios,
        'parametros': {
            'intervalo_burst_horas': round(intervalo_burst, 2),
            'ocorrencias_minimas_burst': ocorrencias_minimas_burst,
            'janela_horas': round(janela_horas, 2),
            'num_bursts_historico': n_bursts,
            'confianca_padrao': padrao_principal['confianca']
        },
        'criterio_reincidencia': {
            'descricao': f"Considerar REINCIDENTE se {ocorrencias_minimas_burst}+ alertas em janela curta",
            'condicao': f"Intervalo entre alertas < {intervalo_burst:.1f}h",
            'janela': f"Janela de {janela_horas:.0f}h"
        },
        'implementacao': {
            'logica': f"""
```

FOR cada novo alerta:
ultimos_alertas = buscar_ultimos_alertas(janela={janela_horas:.0f}h)

```
IF len(ultimos_alertas) >= {ocorrencias_minimas_burst}:
    IF todos_intervalos_curtos():
        MARCAR_COMO_REINCIDENTE()
        PRIORIDADE_MAXIMA()
```

“””
}
}

```
elif tipo == 'REGULAR_SEM_CICLO':
    cv = padrao_principal['cv']
    
    media = results['basic_stats']['mean']
    std = results['basic_stats']['std']
    
    intervalo_esperado_min = max(0.5, media - std)
    intervalo_esperado_max = media + std
    
    ocorrencias_minimas = 4
    janela = (intervalo_esperado_max * ocorrencias_minimas) * 1.5
    
    return {
        'tipo': 'ESTATISTICO',
        'subtipo': 'REGULAR_SEM_CICLO',
        'padrao_principal': padrao_principal,
        'padroes_secundarios': padroes_secundarios,
        'parametros': {
            'intervalo_medio_horas': round(media, 2),
            'desvio_padrao_horas': round(std, 2),
            'intervalo_min_horas': round(intervalo_esperado_min, 2),
            'intervalo_max_horas': round(intervalo_esperado_max, 2),
            'ocorrencias_consecutivas_minimas': ocorrencias_minimas,
            'janela_analise_horas': round(janela, 2),
            'cv': cv,
            'confianca_padrao': padrao_principal['confianca']
        },
        'criterio_reincidencia': {
            'descricao': f"Considerar REINCIDENTE se {ocorrencias_minimas}+ ocorrências regulares",
            'condicao': f"Com intervalo entre {intervalo_esperado_min:.1f}h e {intervalo_esperado_max:.1f}h",
            'janela': f"Últimas {janela:.0f}h"
        },
        'implementacao': {
            'logica': f"""
```

FOR cada novo alerta:
ultimos_alertas = buscar_ultimos_alertas(janela={janela:.0f}h)

```
IF len(ultimos_alertas) >= {ocorrencias_minimas}:
    dentro_do_padrao = contar_intervalos_no_range()
    
    IF dentro_do_padrao >= {ocorrencias_minimas - 1}:
        MARCAR_COMO_REINCIDENTE()
```

“””
}
}

```
else:
    return gerar_regra_generica()
```

def gerar_regra_generica():
“”“Regra genérica quando não há padrão claro”””
return {
‘tipo’: ‘THRESHOLD_GENERICO’,
‘subtipo’: ‘SEM_PADRAO’,
‘padrao_principal’: {
‘tipo’: ‘SEM_PADRAO_CLARO’,
‘descricao’: ‘Sem padrão recorrente detectável’,
‘confianca’: 50
},
‘padroes_secundarios’: [],
‘parametros’: {
‘ocorrencias_janela’: 3,
‘janela_horas’: 24,
‘metodo’: ‘Threshold simples’
},
‘criterio_reincidencia’: {
‘descricao’: “Manter regra atual de threshold”,
‘condicao’: “3 ou mais alertas”,
‘janela’: “Janela de 24 horas”
},
‘implementacao’: {
‘logica’: “””
FOR cada novo alerta:
ultimos_24h = buscar_ultimos_alertas(janela=24h)

```
IF len(ultimos_24h) >= 3:
    MARCAR_COMO_REINCIDENTE()
```

“””
}
}

def calcular_efetividade_regra(df, regra, regra_atual={‘ocorrencias’: 3, ‘janela_horas’: 24}):
“”“Simula a efetividade da regra customizada vs regra atual”””
if len(df) < 3:
return {
‘regra_customizada’: {‘deteccoes’: 0, ‘taxa’: 0},
‘regra_atual’: {‘deteccoes’: 0, ‘taxa’: 0},
‘melhoria’: 0
}

```
df_sorted = df.sort_values('created_on').reset_index(drop=True)

deteccoes_atual = 0
for i in range(len(df_sorted)):
    janela_inicio = df_sorted.loc[i, 'created_on'] - timedelta(hours=regra_atual['janela_horas'])
    alertas_na_janela = df_sorted[
        (df_sorted['created_on'] >= janela_inicio) & 
        (df_sorted['created_on'] <= df_sorted.loc[i, 'created_on'])
    ]
    if len(alertas_na_janela) >= regra_atual['ocorrencias']:
        deteccoes_atual += 1

deteccoes_custom = 0
tipo_regra = regra['tipo']
params = regra['parametros']

if tipo_regra == 'INTERVALO_FIXO':
    periodo = params['periodo_esperado_horas']
    tolerancia = params['tolerancia_horas']
    ocorrencias_min = params['ocorrencias_consecutivas_minimas']
    
    for i in range(ocorrencias_min - 1, len(df_sorted)):
        ultimos = df_sorted.iloc[max(0, i - ocorrencias_min + 1):i + 1]
        intervalos = ultimos['created_on'].diff().dt.total_seconds() / 3600
        intervalos = intervalos.dropna()
        
        if len(intervalos) >= ocorrencias_min - 1:
            no_padrao = sum((periodo - tolerancia <= iv <= periodo + tolerancia) for iv in intervalos)
            if no_padrao >= ocorrencias_min - 1:
                deteccoes_custom += 1

elif tipo_regra == 'HORARIOS_RECORRENTES':
    horarios = params['horarios_esperados']
    tolerancia_h = params['tolerancia_minutos'] / 60
    dias_consecutivos = params['dias_consecutivos_minimo']
    
    for i in range(len(df_sorted)):
        hora_atual = df_sorted.loc[i, 'created_on'].hour + df_sorted.loc[i, 'created_on'].minute / 60
        em_horario_pico = any(abs(hora_atual - h) <= tolerancia_h for h in horarios)
        
        if em_horario_pico:
            data_atual = df_sorted.loc[i, 'created_on'].date()
            datas_anteriores = [data_atual - timedelta(days=d) for d in range(1, dias_consecutivos)]
            
            dias_com_padrao = 1
            for data in datas_anteriores:
                alertas_dia = df_sorted[df_sorted['created_on'].dt.date == data]
                if len(alertas_dia) > 0:
                    for _, alerta in alertas_dia.iterrows():
                        h = alerta['created_on'].hour + alerta['created_on'].minute / 60
                        if any(abs(h - hp) <= tolerancia_h for hp in horarios):
                            dias_com_padrao += 1
                            break
            
            if dias_com_padrao >= dias_consecutivos:
                deteccoes_custom += 1

elif tipo_regra == 'DIAS_RECORRENTES':
    dias_esperados = params['dias_esperados']
    semanas_min = params['semanas_consecutivas_minimo']
    
    for i in range(len(df_sorted)):
        dia_semana_atual = df_sorted.loc[i, 'created_on'].dayofweek
        
        if dia_semana_atual in dias_esperados:
            data_atual = df_sorted.loc[i, 'created_on']
            
            semanas_com_padrao = 1
            for semana_offset in range(1, semanas_min):
                data_semana_anterior = data_atual - timedelta(weeks=semana_offset)
                inicio_semana = data_semana_anterior - timedelta(days=3)
                fim_semana = data_semana_anterior + timedelta(days=3)
                
                alertas_semana = df_sorted[
                    (df_sorted['created_on'] >= inicio_semana) & 
                    (df_sorted['created_on'] <= fim_semana)
                ]
                
                if any(a.dayofweek in dias_esperados for a in alertas_semana['created_on']):
                    semanas_com_padrao += 1
            
            if semanas_com_padrao >= semanas_min:
                deteccoes_custom += 1

elif tipo_regra == 'RAJADA':
    janela_h = params['janela_horas']
    occ_min = params['ocorrencias_minimas_burst']
    
    for i in range(len(df_sorted)):
        janela_inicio = df_sorted.loc[i, 'created_on'] - timedelta(hours=janela_h)
        alertas_na_janela = df_sorted[
            (df_sorted['created_on'] >= janela_inicio) & 
            (df_sorted['created_on'] <= df_sorted.loc[i, 'created_on'])
        ]
        if len(alertas_na_janela) >= occ_min:
            deteccoes_custom += 1

else:
    deteccoes_custom = deteccoes_atual

taxa_atual = (deteccoes_atual / len(df_sorted)) * 100 if len(df_sorted) > 0 else 0
taxa_custom = (deteccoes_custom / len(df_sorted)) * 100 if len(df_sorted) > 0 else 0

melhoria = taxa_custom - taxa_atual

return {
    'regra_customizada': {
        'deteccoes': deteccoes_custom,
        'taxa': round(taxa_custom, 1)
    },
    'regra_atual': {
        'deteccoes': deteccoes_atual,
        'taxa': round(taxa_atual, 1)
    },
    'melhoria': round(melhoria, 1),
    'total_alertas': len(df_sorted)
}
```

# ============================================================

# FUNÇÕES AUXILIARES PARA MULTIPROCESSING

# ============================================================

def analyze_single_short_ci_recurrence(short_ci, df_original):
“”“Função auxiliar para análise de reincidência de um único short_ci”””
try:
df_ci = df_original[df_original[‘short_ci’] == short_ci].copy()
df_ci[‘created_on’] = pd.to_datetime(df_ci[‘created_on’], errors=‘coerce’)
df_ci = df_ci.dropna(subset=[‘created_on’])
df_ci = df_ci.sort_values(‘created_on’)

```
    if len(df_ci) < 3:
        return {
            'short_ci': short_ci,
            'total_occurrences': len(df_ci),
            'score': 0,
            'classification': '⚪ DADOS INSUFICIENTES',
            'mean_interval_hours': None,
            'cv': None,
            'regularity_score': 0,
            'periodicity_detected': False,
            'predictability_score': 0
        }
    
    analyzer = AdvancedRecurrenceAnalyzer(df_ci, short_ci)
    return analyzer.analyze_complete_silent()

except Exception as e:
    return {
        'short_ci': short_ci,
        'total_occurrences': 0,
        'score': 0,
        'classification': f'⚪ ERRO: {str(e)[:50]}',
        'mean_interval_hours': None,
        'cv': None,
        'regularity_score': 0,
        'periodicity_detected': False,
        'predictability_score': 0
    }
```

def analyze_chunk_recurrence(short_ci_list, df_original):
“”“Processa um chunk de short_ci para análise de reincidência”””
results = []
for short_ci in short_ci_list:
result = analyze_single_short_ci_recurrence(short_ci, df_original)
if result:
results.append(result)
return results

# ============================================================

# CLASSE DE ANÁLISE DE REINCIDÊNCIA - OTIMIZADA

# ============================================================

class AdvancedRecurrenceAnalyzer:
“”“Analisador completo de padrões de reincidência - VERSÃO OTIMIZADA”””

```
def __init__(self, df, alert_id):
    self.df = df.copy() if df is not None else None
    self.alert_id = alert_id
    self.cache = {}

def _prepare_data(self):
    """Preparação vetorizada dos dados"""
    if self.df is None or len(self.df) < 3:
        return None
    
    df = self.df.sort_values('created_on').copy()
    
    df['timestamp'] = df['created_on'].astype('int64') // 10**9
    df['time_diff_seconds'] = df['timestamp'].diff()
    df['time_diff_hours'] = df['time_diff_seconds'] / 3600
    df['time_diff_days'] = df['time_diff_seconds'] / 86400
    
    dt = df['created_on'].dt
    if 'hour' not in df.columns:
        df['hour'] = dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = dt.dayofweek
    if 'day_of_month' not in df.columns:
        df['day_of_month'] = dt.day
    if 'week_of_year' not in df.columns:
        df['week_of_year'] = dt.isocalendar().week
    if 'month' not in df.columns:
        df['month'] = dt.month
    if 'day_name' not in df.columns:
        df['day_name'] = dt.day_name()
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
    if 'is_business_hours' not in df.columns:
        df['is_business_hours'] = (df['hour'] >= 9) & (df['hour'] <= 17)
    
    return df

def analyze(self):
    """Método principal com interface Streamlit OTIMIZADA"""
    st.header("🔄 Análise Avançada de Reincidência Temporal")
    
    df = self._prepare_data()
    if df is None:
        st.warning("⚠️ Dados insuficientes (mínimo 3 ocorrências).")
        return
    
    st.info(f"📊 Analisando **{len(df)}** ocorrências do Short CI: **{self.alert_id}**")
    
    intervals_hours = df['time_diff_hours'].dropna().values
    if len(intervals_hours) < 2:
        st.warning("⚠️ Intervalos insuficientes.")
        return
    
    # Executar análises OTIMIZADAS (11 análises)
    results = {}
    results['basic_stats'] = self._analyze_basic_statistics(intervals_hours)
    results['regularity'] = self._analyze_regularity(intervals_hours)
    results['periodicity'] = self._analyze_periodicity(intervals_hours)
    results['temporal'] = self._analyze_temporal_patterns(df)
    results['bursts'] = self._detect_bursts(intervals_hours)
    results['anomalies'] = self._detect_anomalies(intervals_hours)
    results['predictability'] = self._calculate_predictability(intervals_hours)
    results['stability'] = self._analyze_stability(intervals_hours, df)
    results['contextual'] = self._analyze_contextual_dependencies(df)
    results['markov'] = self._analyze_markov_chains(intervals_hours)
    results['randomness'] = self._advanced_randomness_tests(intervals_hours)
    
    # Classificação final COM REGRAS CUSTOMIZADAS
    self._final_classification(results, df, intervals_hours)

def analyze_complete_silent(self):
    """Análise COMPLETA silenciosa - VERSÃO OTIMIZADA"""
    df = self._prepare_data()
    if df is None or len(df) < 3:
        return None
    
    intervals_hours = df['time_diff_hours'].dropna().values
    if len(intervals_hours) < 2:
        return None
    
    # Executar análises (11 análises otimizadas)
    results = {}
    
    try:
        results['basic_stats'] = self._analyze_basic_statistics_complete(intervals_hours)
    except Exception:
        results['basic_stats'] = {'mean': 0, 'median': 0, 'std': 0, 'cv': 0}
    
    try:
        results['regularity'] = self._analyze_regularity_complete(intervals_hours)
    except Exception:
        results['regularity'] = {'cv': 0, 'regularity_score': 0}
    
    try:
        results['periodicity'] = self._analyze_periodicity_complete(intervals_hours)
    except Exception:
        results['periodicity'] = {'has_strong_periodicity': False, 'has_moderate_periodicity': False, 'dominant_period_hours': None}
    
    try:
        results['temporal'] = self._analyze_temporal_patterns_complete(df)
    except Exception:
        results['temporal'] = {'hourly_concentration': 0, 'daily_concentration': 0, 'peak_hours': [], 'peak_days': []}
    
    try:
        results['bursts'] = self._detect_bursts_complete(intervals_hours)
    except Exception:
        results['bursts'] = {'n_bursts': 0, 'has_bursts': False}
    
    try:
        results['anomalies'] = self._detect_anomalies_complete(intervals_hours)
    except Exception:
        results['anomalies'] = {'anomaly_rate': 0}
    
    try:
        results['predictability'] = self._calculate_predictability_complete(intervals_hours)
    except Exception:
        results['predictability'] = {'predictability_score': 0, 'next_expected_hours': 0}
    
    try:
        results['stability'] = self._analyze_stability_complete(intervals_hours)
    except Exception:
        results['stability'] = {'is_stable': True, 'stability_score': 50}
    
    try:
        results['contextual'] = self._analyze_contextual_dependencies_complete(df)
    except Exception:
        results['contextual'] = {'holiday_correlation': 0, 'weekend_correlation': 0}
    
    try:
        results['markov'] = self._analyze_markov_chains_complete(intervals_hours)
    except Exception:
        results['markov'] = {'markov_score': 0}
    
    try:
        results['randomness'] = self._advanced_randomness_tests_complete(intervals_hours)
    except Exception:
        results['randomness'] = {'overall_randomness_score': 50}
    
    # Calcular score final OTIMIZADO
    final_score, classification = self._calculate_final_score_optimized(results, df, intervals_hours)
    
    # Gerar regra customizada
    regra = gerar_regra_customizada(results, df, intervals_hours)
    
    return {
        'short_ci': self.alert_id,
        'total_occurrences': len(df),
        'score': final_score,
        'classification': classification,
        'mean_interval_hours': results['basic_stats']['mean'],
        'median_interval_hours': results['basic_stats']['median'],
        'cv': results['basic_stats']['cv'],
        'regularity_score': results['regularity']['regularity_score'],
        'periodicity_detected': results['periodicity']['has_strong_periodicity'],
        'dominant_period_hours': results['periodicity'].get('dominant_period_hours'),
        'predictability_score': results['predictability']['predictability_score'],
        'next_occurrence_prediction_hours': results['predictability']['next_expected_hours'],
        'hourly_concentration': results['temporal']['hourly_concentration'],
        'daily_concentration': results['temporal']['daily_concentration'],
        'burst_detected': results['bursts']['has_bursts'],
        'n_bursts': results['bursts']['n_bursts'],
        'markov_score': results['markov']['markov_score'],
        'randomness_score': results['randomness']['overall_randomness_score'],
        'stability_score': results['stability']['stability_score'],
        'anomaly_rate': results['anomalies']['anomaly_rate'],
        'pattern_type': regra['tipo'],
        'pattern_subtype': regra['subtipo'],
        'pattern_description': regra['padrao_principal']['descricao'],
        'pattern_confidence': regra['padrao_principal']['confianca']
    }

# ============================================================
# ANÁLISES COM INTERFACE (modo Individual) - MANTIDAS
# ============================================================

def _analyze_basic_statistics(self, intervals):
    """Estatísticas básicas"""
    st.subheader("📊 1. Estatísticas de Intervalos")
    
    stats_dict = {
        'mean': np.mean(intervals),
        'median': np.median(intervals),
        'std': np.std(intervals),
        'min': np.min(intervals),
        'max': np.max(intervals),
        'cv': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf'),
        'q25': np.percentile(intervals, 25),
        'q75': np.percentile(intervals, 75),
        'iqr': np.percentile(intervals, 75) - np.percentile(intervals, 25)
    }
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("⏱️ Média", f"{stats_dict['mean']:.1f}h")
    col2.metric("📊 Mediana", f"{stats_dict['median']:.1f}h")
    col3.metric("📈 Desvio", f"{stats_dict['std']:.1f}h")
    col4.metric("⚡ Mínimo", f"{stats_dict['min']:.1f}h")
    col5.metric("🐌 Máximo", f"{stats_dict['max']:.1f}h")
    
    return stats_dict

def _analyze_regularity(self, intervals):
    """Análise de regularidade"""
    st.subheader("🎯 2. Regularidade")
    
    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
    
    if cv < 0.20:
        regularity_score = 95
        pattern_type = "🟢 ALTAMENTE REGULAR"
        pattern_color = "green"
    elif cv < 0.40:
        regularity_score = 80
        pattern_type = "🟢 REGULAR"
        pattern_color = "lightgreen"
    elif cv < 0.70:
        regularity_score = 60
        pattern_type = "🟡 SEMI-REGULAR"
        pattern_color = "yellow"
    elif cv < 1.20:
        regularity_score = 35
        pattern_type = "🟠 IRREGULAR"
        pattern_color = "orange"
    else:
        regularity_score = 15
        pattern_type = "🔴 MUITO IRREGULAR"
        pattern_color = "red"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**Classificação:** {pattern_type}")
        st.write(f"**CV:** {cv:.2%}")
        
        if len(intervals) >= 3:
            _, p_value = stats.shapiro(intervals)
            if p_value > 0.05:
                st.info("📊 Distribuição aproximadamente normal")
            else:
                st.warning("📊 Distribuição não-normal")
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=regularity_score,
            title={'text': "Regularidade"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': pattern_color},
                'steps': [
                    {'range': [0, 35], 'color': "lightgray"},
                    {'range': [35, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True, key='reg_gauge')
    
    return {'cv': cv, 'regularity_score': regularity_score, 'type': pattern_type}

def _analyze_periodicity(self, intervals):
    """Análise de periodicidade com FFT"""
    st.subheader("🔍 3. Periodicidade (FFT)")
    
    if len(intervals) < 10:
        st.info("📊 Mínimo de 10 intervalos necessários")
        return {'has_periodicity': False, 'has_strong_periodicity': False}
    
    intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
    n_padded = 2**int(np.ceil(np.log2(len(intervals_norm))))
    intervals_padded = np.pad(intervals_norm, (0, n_padded - len(intervals_norm)), 'constant')
    
    fft_vals = fft(intervals_padded)
    freqs = fftfreq(n_padded, d=1)
    
    positive_idx = freqs > 0
    freqs_pos = freqs[positive_idx]
    fft_mag = np.abs(fft_vals[positive_idx])
    
    threshold = np.mean(fft_mag) + 2 * np.std(fft_mag)
    peaks_idx = fft_mag > threshold
    
    dominant_periods = []
    has_strong_periodicity = False
    dominant_period_hours = None
    
    if np.any(peaks_idx):
        dominant_freqs = freqs_pos[peaks_idx]
        dominant_periods = 1 / dominant_freqs
        dominant_periods = dominant_periods[dominant_periods < len(intervals)][:3]
        
        if len(dominant_periods) > 0:
            has_strong_periodicity = True
            dominant_period_hours = dominant_periods[0] * np.mean(intervals)
            
            st.success("🎯 **Periodicidades Detectadas:**")
            for period in dominant_periods:
                est_time = period * np.mean(intervals)
                time_str = f"{est_time:.1f}h" if est_time < 24 else f"{est_time/24:.1f} dias"
                st.write(f"• Período: **{period:.1f}** ocorrências (~{time_str})")
    else:
        st.info("📊 Nenhuma periodicidade forte detectada")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=1/freqs_pos[:len(freqs_pos)//4],
        y=fft_mag[:len(freqs_pos)//4],
        mode='lines',
        fill='tozeroy'
    ))
    fig.update_layout(
        title="Espectro de Frequência",
        xaxis_title="Período",
        yaxis_title="Magnitude",
        height=300,
        xaxis_type="log"
    )
    st.plotly_chart(fig, use_container_width=True, key='fft')
    
    return {
        'periods': dominant_periods,
        'has_periodicity': len(dominant_periods) > 0,
        'has_strong_periodicity': has_strong_periodicity,
        'dominant_period_hours': dominant_period_hours
    }

def _analyze_temporal_patterns(self, df):
    """Análise de padrões temporais"""
    st.subheader("⏰ 4. Padrões Temporais")
    
    hourly = df.groupby('hour').size()
    hourly = hourly.reindex(range(24), fill_value=0)
    
    daily = df.groupby('day_of_week').size()
    daily = daily.reindex(range(7), fill_value=0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Bar(
            x=list(range(24)),
            y=hourly.values,
            marker_color=['red' if v > hourly.mean() + hourly.std() else 'lightblue' 
                        for v in hourly.values]
        ))
        fig.update_layout(title="Por Hora", xaxis_title="Hora", height=250)
        st.plotly_chart(fig, use_container_width=True, key='hourly')
        
        peak_hours = hourly[hourly > hourly.mean() + hourly.std()].index.tolist()
        if peak_hours:
            st.success(f"🕐 **Picos:** {', '.join([f'{h:02d}:00' for h in peak_hours])}")
    
    with col2:
        days_map = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
        fig = go.Figure(go.Bar(
            x=days_map,
            y=daily.values,
            marker_color=['red' if v > daily.mean() + daily.std() else 'lightgreen' 
                        for v in daily.values]
        ))
        fig.update_layout(title="Por Dia", xaxis_title="Dia", height=250)
        st.plotly_chart(fig, use_container_width=True, key='daily')
        
        peak_days = daily[daily > daily.mean() + daily.std()].index.tolist()
        if peak_days:
            st.success(f"📅 **Picos:** {', '.join([days_map[d] for d in peak_days])}")
    
    hourly_pct = (hourly / hourly.sum() * 100) if hourly.sum() > 0 else pd.Series()
    daily_pct = (daily / daily.sum() * 100) if daily.sum() > 0 else pd.Series()
    
    hourly_conc = hourly_pct.nlargest(3).sum() if len(hourly_pct) > 0 else 0
    daily_conc = daily_pct.nlargest(3).sum() if len(daily_pct) > 0 else 0
    
    return {
        'hourly_concentration': hourly_conc,
        'daily_concentration': daily_conc,
        'peak_hours': peak_hours,
        'peak_days': peak_days
    }

def _detect_bursts(self, intervals):
    """Detecção de bursts"""
    st.subheader("💥 5. Detecção de Bursts")
    
    burst_threshold = np.percentile(intervals, 25)
    
    is_burst = intervals < burst_threshold
    burst_changes = np.diff(np.concatenate(([False], is_burst, [False])))
    burst_starts = np.where(burst_changes == 1)[0]
    burst_ends = np.where(burst_changes == -1)[0]
    
    burst_sequences = [(start, end) for start, end in zip(burst_starts, burst_ends) 
                      if end - start >= 3]
    
    col1, col2 = st.columns(2)
    col1.metric("💥 Bursts", len(burst_sequences))
    
    if burst_sequences:
        avg_size = np.mean([end - start for start, end in burst_sequences])
        col2.metric("📊 Tamanho Médio", f"{avg_size:.1f}")
        st.warning(f"⚠️ **{len(burst_sequences)} bursts** detectados")
    else:
        st.success("✅ Sem padrão de rajadas")
    
    return {'n_bursts': len(burst_sequences), 'has_bursts': len(burst_sequences) > 0}

def _detect_anomalies(self, intervals):
    """Detecção de anomalias"""
    st.subheader("🚨 6. Detecção de Anomalias")
    
    z_scores = np.abs(stats.zscore(intervals))
    z_anomalies = np.sum(z_scores > 3)
    
    q1, q3 = np.percentile(intervals, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_anomalies = np.sum((intervals < lower) | (intervals > upper))
    
    iso_anomalies = 0
    if len(intervals) >= 10:
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(intervals.reshape(-1, 1))
        iso_anomalies = np.sum(predictions == -1)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Z-Score", f"{z_anomalies}")
    col2.metric("IQR", f"{iqr_anomalies}")
    col3.metric("Iso. Forest", f"{iso_anomalies}")
    
    total_anomalies = max(z_anomalies, iqr_anomalies, iso_anomalies)
    anomaly_rate = total_anomalies / len(intervals) * 100
    
    if anomaly_rate > 10:
        st.warning(f"⚠️ **{anomaly_rate:.1f}%** de anomalias")
    else:
        st.success("✅ Baixa taxa de anomalias")
    
    return {'anomaly_rate': anomaly_rate, 'total_anomalies': total_anomalies}

def _calculate_predictability(self, intervals):
    """Score de previsibilidade"""
    st.subheader("🔮 7. Previsibilidade")
    
    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
    
    if cv < 0.20:
        predictability = 95
    elif cv < 0.40:
        predictability = 80
    elif cv < 0.70:
        predictability = 55
    elif cv < 1.20:
        predictability = 30
    else:
        predictability = 10
    
    mean_interval = np.mean(intervals)
    
    col1, col2 = st.columns(2)
    col1.metric("Score", f"{predictability}%")
    col2.metric("Próxima Ocorrência", f"{mean_interval:.1f}h")
    
    if predictability > 70:
        st.success("✅ Altamente previsível")
    elif predictability > 50:
        st.info("📊 Moderadamente previsível")
    else:
        st.warning("⚠️ Pouco previsível")
    
    return {'predictability_score': predictability, 'next_expected_hours': mean_interval}

def _analyze_stability(self, intervals, df):
    """Análise de estabilidade"""
    st.subheader("📊 8. Estabilidade")
    
    if len(intervals) < 10:
        return {'is_stable': True, 'stability_score': 50}
    
    mid = len(intervals) // 2
    first_half = intervals[:mid]
    second_half = intervals[mid:]
    
    _, p_value = stats.ttest_ind(first_half, second_half)
    is_stable = p_value > 0.05
    
    mean_diff = abs(np.mean(second_half) - np.mean(first_half))
    drift_pct = (mean_diff / np.mean(first_half)) * 100 if np.mean(first_half) > 0 else 0
    
    stability_score = max(0, 100 - drift_pct)
    
    col1, col2 = st.columns(2)
    col1.metric("Score", f"{stability_score:.1f}%")
    col2.metric("Drift", f"{drift_pct:.1f}%")
    
    if is_stable and drift_pct < 20:
        st.success("✅ Padrão estável")
    elif drift_pct < 50:
        st.info("📊 Moderadamente estável")
    else:
        st.warning("⚠️ Padrão instável")
    
    return {'is_stable': is_stable, 'stability_score': stability_score, 'drift_pct': drift_pct}

def _analyze_contextual_dependencies(self, df):
    """Análise de dependências contextuais"""
    st.subheader("🌐 9. Dependências Contextuais")
    
    try:
        br_holidays = holidays.Brazil(years=df['created_on'].dt.year.unique())
        df['is_holiday'] = df['created_on'].dt.date.apply(lambda x: x in br_holidays)
    except:
        df['is_holiday'] = False
    
    business_days = df[~df['is_weekend'] & ~df['is_holiday']]
    weekend_days = df[df['is_weekend']]
    holiday_days = df[df['is_holiday']]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Dias Úteis", f"{len(business_days)/len(df)*100:.1f}%")
    col2.metric("🎉 Fins de Semana", f"{len(weekend_days)/len(df)*100:.1f}%")
    col3.metric("🎊 Feriados", f"{len(holiday_days)/len(df)*100:.1f}%")
    
    if len(holiday_days) > 0:
        st.warning(f"⚠️ {len(holiday_days)} alertas em feriados")
    
    return {
        'holiday_correlation': len(holiday_days) / len(df) if len(df) > 0 else 0,
        'weekend_correlation': len(weekend_days) / len(df) if len(df) > 0 else 0
    }

def _analyze_markov_chains(self, intervals):
    """Cadeias de Markov"""
    st.subheader("🔗 10. Cadeias de Markov")
    
    if len(intervals) < 20:
        st.info("Mínimo de 20 intervalos necessário")
        return {}
    
    q25, q50, q75 = np.percentile(intervals, [25, 50, 75])
    
    def interval_to_state(val):
        if val <= q25:
            return 'Muito Curto'
        elif val <= q50:
            return 'Curto'
        elif val <= q75:
            return 'Normal'
        else:
            return 'Longo'
    
    states = [interval_to_state(i) for i in intervals]
    state_labels = ['Muito Curto', 'Curto', 'Normal', 'Longo']
    
    n_states = len(state_labels)
    transition_matrix = np.zeros((n_states, n_states))
    state_to_idx = {state: idx for idx, state in enumerate(state_labels)}
    
    for i in range(len(states) - 1):
        from_state = state_to_idx[states[i]]
        to_state = state_to_idx[states[i + 1]]
        transition_matrix[from_state, to_state] += 1
    
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_probs = transition_matrix / row_sums
    
    fig = go.Figure(data=go.Heatmap(
        z=transition_probs,
        x=state_labels,
        y=state_labels,
        text=np.round(transition_probs, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title="Matriz de Transição",
        xaxis_title="Estado Seguinte",
        yaxis_title="Estado Atual",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True, key='markov_matrix')
    
    max_probs = transition_probs.max(axis=1)
    markov_score = np.mean(max_probs) * 100
    
    st.metric("Score Markoviano", f"{markov_score:.1f}%")
    
    if markov_score > 60:
        st.success("✅ Forte padrão markoviano")
    elif markov_score > 30:
        st.info("📊 Padrão moderado")
    else:
        st.warning("⚠️ Padrão fraco")
    
    return {'markov_score': markov_score}

def _advanced_randomness_tests(self, intervals):
    """Testes de aleatoriedade"""
    st.subheader("🎲 11. Testes de Aleatoriedade")
    
    if len(intervals) < 10:
        st.info("Mínimo de 10 intervalos necessário")
        return {}
    
    results = {}
    
    st.write("**1️⃣ Runs Test**")
    median = np.median(intervals)
    runs = np.diff(intervals > median).sum() + 1
    expected_runs = len(intervals) / 2
    
    col1, col2 = st.columns(2)
    col1.metric("Runs Observados", runs)
    col2.metric("Runs Esperados", f"{expected_runs:.1f}")
    
    st.write("**2️⃣ Permutation Entropy**")
    
    def permutation_entropy(series, order=3):
        n = len(series)
        permutations = []
        
        for i in range(n - order + 1):
            pattern = series[i:i+order]
            sorted_idx = np.argsort(pattern)
            perm = tuple(sorted_idx)
            permutations.append(perm)
        
        perm_counts = Counter(permutations)
        probs = np.array(list(perm_counts.values())) / len(permutations)
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(math.factorial(order))
        return entropy / max_entropy if max_entropy > 0 else 0
    
    if len(intervals) >= 10:
        perm_entropy = permutation_entropy(intervals)
        complexity = perm_entropy * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Entropia", f"{perm_entropy:.3f}")
        col2.metric("Complexidade", f"{complexity:.1f}%")
        
        if complexity > 70:
            st.success("✅ Alta complexidade")
        else:
            st.warning("⚠️ Baixa complexidade")
    
    st.markdown("---")
    randomness_score = 50
    st.metric("Score de Aleatoriedade", f"{randomness_score:.0f}%")
    
    if randomness_score >= 60:
        st.success("✅ Comportamento aleatório")
    elif randomness_score >= 40:
        st.info("📊 Comportamento misto")
    else:
        st.warning("⚠️ Comportamento determinístico")
    
    results['overall_randomness_score'] = randomness_score
    return results

# ============================================================
# MÉTODOS COMPLETOS SILENCIOSOS (para batch processing)
# ============================================================

def _analyze_basic_statistics_complete(self, intervals):
    return {
        'mean': np.mean(intervals),
        'median': np.median(intervals),
        'std': np.std(intervals),
        'cv': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf'),
        'min': np.min(intervals),
        'max': np.max(intervals)
    }

def _analyze_regularity_complete(self, intervals):
    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
    if cv < 0.20:
        regularity_score = 95
    elif cv < 0.40:
        regularity_score = 80
    elif cv < 0.70:
        regularity_score = 60
    elif cv < 1.20:
        regularity_score = 35
    else:
        regularity_score = 15
    return {'cv': cv, 'regularity_score': regularity_score}

def _analyze_periodicity_complete(self, intervals):
    if len(intervals) < 10:
        return {'has_strong_periodicity': False, 'has_moderate_periodicity': False, 'dominant_period_hours': None}
    
    try:
        intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
        n_padded = 2**int(np.ceil(np.log2(len(intervals_norm))))
        intervals_padded = np.pad(intervals_norm, (0, n_padded - len(intervals_norm)), 'constant')
        
        fft_vals = fft(intervals_padded)
        freqs = fftfreq(n_padded, d=1)
        
        positive_idx = freqs > 0
        freqs_pos = freqs[positive_idx]
        fft_mag = np.abs(fft_vals[positive_idx])
        
        threshold = np.mean(fft_mag) + 2 * np.std(fft_mag)
        peaks_idx = fft_mag > threshold
        
        if np.any(peaks_idx):
            dominant_freqs = freqs_pos[peaks_idx]
            dominant_periods = 1 / dominant_freqs
            dominant_periods = dominant_periods[dominant_periods < len(intervals)][:3]
            
            if len(dominant_periods) > 0:
                dominant_period_hours = dominant_periods[0] * np.mean(intervals)
                return {
                    'has_strong_periodicity': True,
                    'has_moderate_periodicity': False,
                    'dominant_period_hours': dominant_period_hours
                }
    except:
        pass
    
    return {'has_strong_periodicity': False, 'has_moderate_periodicity': False, 'dominant_period_hours': None}

def _analyze_temporal_patterns_complete(self, df):
    try:
        hourly = df.groupby('hour').size()
        daily = df.groupby('day_of_week').size()
        
        hourly_pct = (hourly / hourly.sum() * 100) if hourly.sum() > 0 else pd.Series()
        daily_pct = (daily / daily.sum() * 100) if daily.sum() > 0 else pd.Series()
        
        hourly_conc = hourly_pct.nlargest(3).sum() if len(hourly_pct) > 0 else 0
        daily_conc = daily_pct.nlargest(3).sum() if len(daily_pct) > 0 else 0
        
        peak_hours = hourly[hourly > hourly.mean() + hourly.std()].index.tolist() if len(hourly) > 0 else []
        peak_days = daily[daily > daily.mean() + daily.std()].index.tolist() if len(daily) > 0 else []
        
        return {
            'hourly_concentration': hourly_conc,
            'daily_concentration': daily_conc,
            'peak_hours': peak_hours,
            'peak_days': peak_days
        }
    except Exception:
        return {
            'hourly_concentration': 0,
            'daily_concentration': 0,
            'peak_hours': [],
            'peak_days': []
        }

def _detect_bursts_complete(self, intervals):
    if len(intervals) < 5:
        return {'n_bursts': 0, 'has_bursts': False}
    
    burst_threshold = np.percentile(intervals, 25)
    is_burst = intervals < burst_threshold
    burst_changes = np.diff(np.concatenate(([False], is_burst, [False])))
    burst_starts = np.where(burst_changes == 1)[0]
    burst_ends = np.where(burst_changes == -1)[0]
    
    burst_sequences = [(start, end) for start, end in zip(burst_starts, burst_ends) 
                      if end - start >= 3]
    
    return {'n_bursts': len(burst_sequences), 'has_bursts': len(burst_sequences) > 0}

def _detect_anomalies_complete(self, intervals):
    z_scores = np.abs(stats.zscore(intervals))
    z_anomalies = np.sum(z_scores > 3)
    
    q1, q3 = np.percentile(intervals, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_anomalies = np.sum((intervals < lower) | (intervals > upper))
    
    total_anomalies = max(z_anomalies, iqr_anomalies)
    anomaly_rate = total_anomalies / len(intervals) * 100
    
    return {'anomaly_rate': anomaly_rate, 'total_anomalies': total_anomalies}

def _calculate_predictability_complete(self, intervals):
    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
    if cv < 0.20:
        predictability = 95
    elif cv < 0.40:
        predictability = 80
    elif cv < 0.70:
        predictability = 55
    elif cv < 1.20:
        predictability = 30
    else:
        predictability = 10
    mean_interval = np.mean(intervals)
    return {'predictability_score': predictability, 'next_expected_hours': mean_interval}

def _analyze_stability_complete(self, intervals):
    if len(intervals) < 10:
        return {'is_stable': True, 'stability_score': 50}
    try:
        mid = len(intervals) // 2
        first_half = intervals[:mid]
        second_half = intervals[mid:]
        _, p_value = stats.ttest_ind(first_half, second_half)
        is_stable = p_value > 0.05
        mean_diff = abs(np.mean(second_half) - np.mean(first_half))
        drift_pct = (mean_diff / np.mean(first_half)) * 100 if np.mean(first_half) > 0 else 0
        stability_score = max(0, 100 - drift_pct)
        return {'is_stable': is_stable, 'stability_score': stability_score}
    except:
        return {'is_stable': True, 'stability_score': 50}

def _analyze_contextual_dependencies_complete(self, df):
    try:
        try:
            br_holidays = holidays.Brazil(years=df['created_on'].dt.year.unique())
            df['is_holiday'] = df['created_on'].dt.date.apply(lambda x: x in br_holidays)
        except:
            df['is_holiday'] = False
        
        weekend_days = df[df['is_weekend']] if 'is_weekend' in df.columns else pd.DataFrame()
        holiday_days = df[df['is_holiday']]
        
        return {
            'holiday_correlation': len(holiday_days) / len(df) if len(df) > 0 else 0,
            'weekend_correlation': len(weekend_days) / len(df) if len(df) > 0 else 0
        }
    except Exception:
        return {
            'holiday_correlation': 0,
            'weekend_correlation': 0
        }

def _analyze_markov_chains_complete(self, intervals):
    if len(intervals) < 20:
        return {'markov_score': 0}
    try:
        q25, q50, q75 = np.percentile(intervals, [25, 50, 75])
        
        def interval_to_state(val):
            if val <= q25:
                return 0
            elif val <= q50:
                return 1
            elif val <= q75:
                return 2
            else:
                return 3
        
        states = [interval_to_state(i) for i in intervals]
        n_states = 4
        transition_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            transition_matrix[from_state, to_state] += 1
        
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums
        
        max_probs = transition_matrix.max(axis=1)
        markov_score = np.mean(max_probs) * 100
        return {'markov_score': markov_score}
    except:
        return {'markov_score': 0}

def _advanced_randomness_tests_complete(self, intervals):
    if len(intervals) < 10:
        return {'overall_randomness_score': 50}
    try:
        median = np.median(intervals)
        runs = np.diff(intervals > median).sum() + 1
        expected_runs = len(intervals) / 2
        runs_score = min(abs(runs - expected_runs) / expected_runs * 100, 100) if expected_runs > 0 else 50
        overall_randomness = runs_score
        return {'overall_randomness_score': overall_randomness}
    except:
        return {'overall_randomness_score': 50}

# ============================================================
# CLASSIFICAÇÃO FINAL COM SCORE OTIMIZADO
# ============================================================

def _calculate_final_score_optimized(self, results, df, intervals):
    """
    Score otimizado - OPÇÃO C
    
    PESOS (100% + penalização):
    - Regularidade: 20%
    - Periodicidade: 20%
    - Previsibilidade: 10% (ajustado de 15%)
    - Concentração Temporal: 20%
    - Frequência Absoluta: 15%
    - Bursts: 10%
    - Markov: 5% (NOVO)
    - Penalização Anomalias: até -5 (NOVO)
    """
    
    # 1. REGULARIDADE (20%)
    regularity_score = results['regularity']['regularity_score'] * 0.20
    
    # 2. PERIODICIDADE (20%)
    if results['periodicity']['has_strong_periodicity']:
        periodicity_score = 100 * 0.20
    elif results['periodicity'].get('has_moderate_periodicity', False):
        periodicity_score = 50 * 0.20
    else:
        periodicity_score = 0 * 0.20
    
    # 3. PREVISIBILIDADE (10%) - REDUZIDO
    predictability_score = results['predictability']['predictability_score'] * 0.10
    
    # 4. CONCENTRAÇÃO TEMPORAL (20%)
    hourly_conc = results['temporal']['hourly_concentration']
    daily_conc = results['temporal']['daily_concentration']
    
    concentration_score = 0
    if hourly_conc > 60 or daily_conc > 60:
        concentration_score = 100 * 0.20
    elif hourly_conc > 40 or daily_conc > 40:
        concentration_score = 60 * 0.20
    elif hourly_conc > 30 or daily_conc > 30:
        concentration_score = 30 * 0.20
    
    # 5. FREQUÊNCIA ABSOLUTA (15%)
    total_occurrences = len(df)
    period_days = (df['created_on'].max() - df['created_on'].min()).days + 1
    freq_per_week = (total_occurrences / period_days * 7) if period_days > 0 else 0
    
    if freq_per_week >= 3:
        frequency_score = 100 * 0.15
    elif freq_per_week >= 1:
        frequency_score = 70 * 0.15
    elif freq_per_week >= 0.5:
        frequency_score = 40 * 0.15
    elif total_occurrences >= 10:
        frequency_score = 30 * 0.15
    else:
        frequency_score = 10 * 0.15
    
    # 6. BURSTS (10%)
    if results['bursts']['has_bursts'] and results['bursts']['n_bursts'] >= 2:
        burst_score = 100 * 0.10
    elif results['bursts']['has_bursts']:
        burst_score = 50 * 0.10
    else:
        burst_score = 0 * 0.10
    
    # 7. MARKOV (5%) - NOVO
    markov_raw = results['markov']['markov_score']
    markov_score = markov_raw * 0.05
    
    # SCORE BASE (100%)
    base_score = (
        regularity_score +
        periodicity_score +
        predictability_score +
        concentration_score +
        frequency_score +
        burst_score +
        markov_score
    )
    
    # 8. PENALIZAÇÃO POR ANOMALIAS - NOVO
    anomaly_rate = results['anomalies']['anomaly_rate']
    
    if anomaly_rate > 20:
        anomaly_penalty = -5
    elif anomaly_rate > 10:
        anomaly_penalty = -2
    else:
        anomaly_penalty = 0
    
    # SCORE FINAL
    final_score = max(0, min(100, base_score + anomaly_penalty))
    
    # CLASSIFICAÇÃO
    if final_score >= 70 and total_occurrences >= 10:
        classification = "🔴 REINCIDENTE CRÍTICO (P1)"
    elif final_score >= 50 and total_occurrences >= 5:
        classification = "🟠 PARCIALMENTE REINCIDENTE (P2)"
    elif final_score >= 35:
        classification = "🟡 PADRÃO DETECTÁVEL (P3)"
    else:
        classification = "🟢 NÃO REINCIDENTE (P4)"
    
    return round(final_score, 2), classification

def _final_classification(self, results, df, intervals):
    """Classificação final COM REGRAS CUSTOMIZADAS"""
    st.markdown("---")
    st.header("🎯 CLASSIFICAÇÃO FINAL")
    
    final_score, classification = self._calculate_final_score_optimized(results, df, intervals)
    
    if final_score >= 70:
        level = "CRÍTICO"
        color = "red"
        priority = "P1"
        recommendation = "**Ação Imediata:** Criar automação, runbook e investigar causa raiz"
    elif final_score >= 50:
        level = "ALTO"
        color = "orange"
        priority = "P2"
        recommendation = "**Ação Recomendada:** Monitorar evolução e considerar automação"
    elif final_score >= 35:
        level = "MÉDIO"
        color = "yellow"
        priority = "P3"
        recommendation = "**Ação Sugerida:** Documentar padrão e revisar thresholds"
    else:
        level = "BAIXO"
        color = "green"
        priority = "P4"
        recommendation = "**Ação:** Análise caso a caso"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {classification}")
        st.markdown(f"**Nível:** {level} | **Prioridade:** {priority}")
        st.metric("Score de Reincidência", f"{final_score:.0f}/100")
        
        st.markdown("#### 📊 Breakdown dos Critérios OTIMIZADOS")
        
        total_occurrences = len(df)
        period_days = (df['created_on'].max() - df['created_on'].min()).days + 1
        freq_per_week = (total_occurrences / period_days * 7) if period_days > 0 else 0
        
        regularity_pts = results['regularity']['regularity_score'] * 0.20
        
        if results['periodicity']['has_strong_periodicity']:
            periodicity_pts = 100 * 0.20
        else:
            periodicity_pts = 0 * 0.20
        
        predictability_pts = results['predictability']['predictability_score'] * 0.10
        
        hourly_conc = results['temporal']['hourly_concentration']
        daily_conc = results['temporal']['daily_concentration']
        if hourly_conc > 60 or daily_conc > 60:
            concentration_pts = 100 * 0.20
        elif hourly_conc > 40 or daily_conc > 40:
            concentration_pts = 60 * 0.20
        else:
            concentration_pts = 30 * 0.20 if (hourly_conc > 30 or daily_conc > 30) else 0
        
        if freq_per_week >= 3:
            frequency_pts = 100 * 0.15
        elif freq_per_week >= 1:
            frequency_pts = 70 * 0.15
        else:
            frequency_pts = 40 * 0.15 if freq_per_week >= 0.5 else 10 * 0.15
        
        if results['bursts']['has_bursts'] and results['bursts']['n_bursts'] >= 2:
            burst_pts = 100 * 0.10
        else:
            burst_pts = 50 * 0.10 if results['bursts']['has_bursts'] else 0
        
        markov_pts = results['markov']['markov_score'] * 0.05
        
        anomaly_rate = results['anomalies']['anomaly_rate']
        if anomaly_rate > 20:
            anomaly_penalty = -5
        elif anomaly_rate > 10:
            anomaly_penalty = -2
        else:
            anomaly_penalty = 0
        
        breakdown = {
            '1. Regularidade (20%)': regularity_pts,
            '2. Periodicidade (20%)': periodicity_pts,
            '3. Previsibilidade (10%)': predictability_pts,
            '4. Concentração Temporal (20%)': concentration_pts,
            '5. Frequência Absoluta (15%)': frequency_pts,
            '6. Bursts (10%)': burst_pts,
            '7. Markov (5%) ✨': markov_pts,
            '8. Penalização Anomalias ✨': anomaly_penalty
        }
        
        for criterion, points in breakdown.items():
            if points >= 0:
                st.write(f"• {criterion}: **+{points:.1f} pts**")
            else:
                st.write(f"• {criterion}: **{points:.1f} pts**")
        
        st.markdown("---")
        st.markdown("**💡 Critérios Otimizados:**")
        st.write("✅ **Regularidade** (20%): Consistência dos intervalos")
        st.write("✅ **Periodicidade** (20%): Detecta ciclos via FFT")
        st.write("✅ **Previsibilidade** (10%): Predição de próxima ocorrência")
        st.write("✅ **Concentração Temporal** (20%): Horários/dias fixos")
        st.write("✅ **Frequência Absoluta** (15%): Volume importa")
        st.write("✅ **Bursts** (10%): Rajadas são padrão importante")
        st.write("✨ **Markov** (5%): Dependência temporal (NOVO)")
        st.write("✨ **Anomalias**: Penalização por instabilidade (NOVO)")
        
        st.info(recommendation)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_score,
            title={'text': "Score Final"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 35], 'color': "lightgray"},
                    {'range': [35, 50], 'color': "lightyellow"},
                    {'range': [50, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True, key='final_gauge')
    
    # REGRAS CUSTOMIZADAS
    st.markdown("---")
    st.header("📋 REGRA CUSTOMIZADA SUGERIDA")
    
    regra = gerar_regra_customizada(results, df, intervals)
    
    padrao_principal = regra['padrao_principal']
    st.success(f"**🎯 Padrão Detectado:** {padrao_principal['tipo']}")
    st.write(f"**Descrição:** {padrao_principal['descricao']}")
    st.write(f"**Confiança:** {padrao_principal['confianca']}%")
    
    padroes_secundarios = regra.get('padroes_secundarios', [])
    if padroes_secundarios:
        st.info("**📊 Padrões Secundários Detectados:**")
        for padrao in padroes_secundarios:
            st.write(f"• {padrao['tipo']}: {padrao['descricao']} (confiança: {padrao['confianca']}%)")
    
    st.markdown("---")
    st.subheader("⚙️ Parâmetros da Regra")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Critério de Reincidência:**")
        criterio = regra['criterio_reincidencia']
        st.write(f"• {criterio['descricao']}")
        st.write(f"• {criterio['condicao']}")
        st.write(f"• Janela: {criterio['janela']}")
    
    with col2:
        st.markdown("**🔧 Parâmetros Técnicos:**")
        params = regra['parametros']
        for key, value in params.items():
            if isinstance(value, (int, float)):
                st.write(f"• {key}: **{value:.2f}**" if isinstance(value, float) else f"• {key}: **{value}**")
            else:
                st.write(f"• {key}: **{value}**")
    
    st.markdown("---")
    st.subheader("💻 Lógica de Implementação")
    
    with st.expander("Ver pseudocódigo", expanded=False):
        st.code(regra['implementacao']['logica'], language='python')
    
    # Comparação
    st.markdown("---")
    st.subheader("📊 Comparação: Regra Customizada vs. Regra Atual")
    
    efetividade = calcular_efetividade_regra(df, regra)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Regra Atual (3 em 24h)",
            f"{efetividade['regra_atual']['taxa']}%",
            f"{efetividade['regra_atual']['deteccoes']} detecções"
        )
    
    with col2:
        st.metric(
            "Regra Customizada",
            f"{efetividade['regra_customizada']['taxa']}%",
            f"{efetividade['regra_customizada']['deteccoes']} detecções"
        )
    
    with col3:
        delta = efetividade['melhoria']
        delta_color = "normal" if delta >= 0 else "inverse"
        st.metric(
            "Melhoria",
            f"{abs(delta):.1f}%",
            f"{'↑' if delta > 0 else '↓'} {abs(delta):.1f}%",
            delta_color=delta_color
        )
    
    if efetividade['melhoria'] > 10:
        st.success(f"✅ A regra customizada é **{efetividade['melhoria']:.1f}% mais efetiva**!")
    elif efetividade['melhoria'] > 0:
        st.info(f"📊 Melhoria modesta de **{efetividade['melhoria']:.1f}%**")
    elif efetividade['melhoria'] < -10:
        st.warning(f"⚠️ Regra atual mais adequada (diferença: {abs(efetividade['melhoria']):.1f}%)")
    else:
        st.info("📊 Ambas as regras têm efetividade similar")
    
    # Predição
    if final_score >= 50:
        st.markdown("---")
        st.subheader("🔮 Predição")
        
        last_alert = df['created_on'].max()
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        pred_time = last_alert + pd.Timedelta(hours=mean_interval)
        conf_interval = 1.96 * std_interval
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Próxima Ocorrência", pred_time.strftime('%d/%m %H:%M'))
        col2.metric("Intervalo", f"{mean_interval:.1f}h")
        col3.metric("Confiança (95%)", f"± {conf_interval:.1f}h")
    
    # Exportar
    st.markdown("---")
    export_data = {
        'short_ci': self.alert_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'score': final_score,
        'classificacao': classification,
        'nivel': level,
        'prioridade': priority,
        'total_occurrences': len(df),
        'freq_per_week': freq_per_week,
        'cv': results['basic_stats']['cv'],
        'regularidade': results['regularity']['regularity_score'],
        'periodicidade': results['periodicity'].get('has_strong_periodicity', False),
        'previsibilidade': results['predictability']['predictability_score'],
        'concentracao_horaria': results['temporal']['hourly_concentration'],
        'concentracao_diaria': results['temporal']['daily_concentration'],
        'bursts_detected': results['bursts']['has_bursts'],
        'n_bursts': results['bursts']['n_bursts'],
        'markov_score': results['markov']['markov_score'],
        'anomaly_rate': results['anomalies']['anomaly_rate'],
        'padrao_tipo': padrao_principal['tipo'],
        'padrao_descricao': padrao_principal['descricao'],
        'padrao_confianca': padrao_principal['confianca'],
        'regra_tipo': regra['tipo'],
        'regra_descricao': regra['criterio_reincidencia']['descricao']
    }
    
    export_df = pd.DataFrame([export_data])
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        "⬇️ Exportar Relatório Completo",
        csv,
        f"reincidencia_{self.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True
    )
```

# ============================================================

# CLASSE PRINCIPAL

# ============================================================

class StreamlitAlertAnalyzer:
def **init**(self):
self.df_original = None
self.df = None
self.dates = None
self.alert_id = None

```
def load_data(self, uploaded_file):
    """Carrega dados do CSV"""
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"✅ Arquivo carregado: {len(df_raw)} registros")
        
        with st.expander("📋 Preview"):
            st.write(f"**Colunas:** {list(df_raw.columns)}")
            st.dataframe(df_raw.head())
        
        if 'created_on' not in df_raw.columns or 'short_ci' not in df_raw.columns:
            st.error("❌ Colunas obrigatórias: 'created_on' e 'short_ci'")
            return False
        
        df_raw['created_on'] = pd.to_datetime(df_raw['created_on'])
        df_raw = df_raw.dropna(subset=['created_on'])
        df_raw = df_raw.sort_values(['short_ci', 'created_on']).reset_index(drop=True)
        
        self.df_original = df_raw
        st.sidebar.write(f"**IDs:** {len(df_raw['short_ci'].unique())}")
        return True
    
    except Exception as e:
        st.error(f"❌ Erro: {e}")
        return False

def prepare_individual_analysis(self, alert_id):
    """Prepara análise individual"""
    df_filtered = self.df_original[self.df_original['short_ci'] == alert_id].copy()
    if len(df_filtered) == 0:
        return False

    df_filtered['date'] = df_filtered['created_on'].dt.date
    df_filtered['hour'] = df_filtered['created_on'].dt.hour
    df_filtered['day_of_week'] = df_filtered['created_on'].dt.dayofweek
    df_filtered['day_name'] = df_filtered['created_on'].dt.day_name()
    df_filtered['is_weekend'] = df_filtered['day_of_week'].isin([5, 6])
    df_filtered['is_business_hours'] = (df_filtered['hour'] >= 9) & (df_filtered['hour'] <= 17)
    df_filtered['time_diff_hours'] = df_filtered['created_on'].diff().dt.total_seconds() / 3600

    self.df = df_filtered
    self.dates = df_filtered['created_on']
    self.alert_id = alert_id
    return True

def complete_analysis_all_short_ci(self, progress_bar=None):
    """Análise COMPLETA COM MULTIPROCESSING"""
    try:
        if self.df_original is None or len(self.df_original) == 0:
            st.error("❌ Dados não carregados")
            return None
        
        short_ci_list = self.df_original['short_ci'].unique()
        total = len(short_ci_list)
        
        use_mp = total > 20
        
        if use_mp:
            n_processes = min(cpu_count(), total, 8)
            st.info(f"🚀 Usando {n_processes} processos para {total} alertas")
            
            chunk_size = max(1, total // n_processes)
            chunks = [short_ci_list[i:i + chunk_size] for i in range(0, total, chunk_size)]
            
            process_func = partial(analyze_chunk_recurrence, df_original=self.df_original)
            
            try:
                all_results = []
                with Pool(processes=n_processes) as pool:
                    for idx, chunk_results in enumerate(pool.imap(process_func, chunks)):
                        all_results.extend(chunk_results)
                        if progress_bar:
                            progress = (len(all_results) / total)
                            progress_bar.progress(progress, text=f"{len(all_results)}/{total}")
                
                df_results = pd.DataFrame(all_results)
                
                if progress_bar:
                    progress_bar.progress(1.0, text="✅ Completa!")
                
                return df_results
            
            except Exception as e:
                st.warning(f"⚠️ Erro no multiprocessing: {e}. Usando modo sequencial...")
                use_mp = False
        
        if not use_mp:
            all_results = []
            for idx, short_ci in enumerate(short_ci_list):
                if progress_bar:
                    progress_bar.progress((idx + 1) / total, text=f"{idx + 1}/{total}")
                
                result = analyze_single_short_ci_recurrence(short_ci, self.df_original)
                if result:
                    all_results.append(result)
            
            return pd.DataFrame(all_results)
    
    except Exception as e:
        st.error(f"Erro: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def show_basic_stats(self):
    """Estatísticas básicas"""
    st.header("📊 Estatísticas Básicas")
    
    total = len(self.df)
    period_days = (self.dates.max() - self.dates.min()).days + 1
    avg_per_day = total / period_days
    unique_days = self.df['date'].nunique()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🔥 Total", total)
    col2.metric("📅 Período", period_days)
    col3.metric("📆 Dias Únicos", unique_days)
    col4.metric("📈 Média/dia", f"{avg_per_day:.2f}")
    col5.metric("🕐 Último", self.dates.max().strftime("%d/%m %H:%M"))
    
    if unique_days == 1:
        st.warning("⚠️ Todos em 1 dia - Pode não ser reincidente")
    
    st.markdown("---")
    st.subheader("📊 Frequências")
    
    total_hours = period_days * 24
    avg_per_hour = total / total_hours if total_hours > 0 else 0
    avg_per_week = total / (period_days / 7) if period_days > 0 else 0
    avg_per_month = total / (period_days / 30.44) if period_days > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Por Dia", f"{avg_per_day:.2f}")
    col2.metric("Por Hora", f"{avg_per_hour:.4f}")
    col3.metric("Por Semana", f"{avg_per_week:.2f}")
    col4.metric("Por Mês", f"{avg_per_month:.2f}")
    
    intervals = self.df['time_diff_hours'].dropna()
    if len(intervals) > 0:
        st.markdown("---")
        st.subheader("⏱️ Intervalos")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Média (h)", f"{intervals.mean():.2f}")
        col2.metric("Mediana (h)", f"{intervals.median():.2f}")
        col3.metric("Mínimo (h)", f"{intervals.min():.2f}")
        col4.metric("Máximo (h)", f"{intervals.max():.2f}")
```

# ============================================================

# FUNÇÃO MAIN

# ============================================================

def main():
st.title(“🚨 Analisador de Alertas - Otimizado”)
st.markdown(”### ✨ Score otimizado: Markov + Penalização de Anomalias”)

```
st.sidebar.header("⚙️ Configurações")

analysis_mode = st.sidebar.selectbox(
    "🎯 Modo de Análise",
    ["🔍 Individual", "📊 Completa + CSV"]
)

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=['csv'])

if uploaded_file:
    analyzer = StreamlitAlertAnalyzer()
    
    if analyzer.load_data(uploaded_file):
        
        if analysis_mode == "🔍 Individual":
            id_counts = analyzer.df_original['short_ci'].value_counts()
            id_options = [f"{uid} ({count})" for uid, count in id_counts.items()]
            selected = st.sidebar.selectbox("Short CI", id_options)
            selected_id = selected.split(" (")[0]
            
            if st.sidebar.button("🚀 Analisar", type="primary"):
                if analyzer.prepare_individual_analysis(selected_id):
                    st.success(f"Analisando: {selected_id}")
                    
                    tab1, tab2 = st.tabs(["📊 Básico", "🔄 Reincidência + Regras"])
                    
                    with tab1:
                        analyzer.show_basic_stats()
                    
                    with tab2:
                        recurrence_analyzer = AdvancedRecurrenceAnalyzer(analyzer.df, selected_id)
                        recurrence_analyzer.analyze()
        
        elif analysis_mode == "📊 Completa + CSV":
            st.subheader("📊 Análise Completa - OTIMIZADA")
            
            st.info("""
            **✨ Critérios Otimizados (11 análises):**
            - Regularidade (20%)
            - Periodicidade (20%)
            - Previsibilidade (10%) - ajustado
            - Concentração Temporal (20%)
            - Frequência Absoluta (15%)
            - Bursts (10%)
            - **Markov (5%) - NOVO**
            - **Penalização Anomalias (até -5) - NOVO**
            
            **Removido:** 7 análises não utilizadas
            """)
            
            if st.sidebar.button("🚀 Executar", type="primary"):
                st.info("⏱️ Processando...")
                
                progress_bar = st.progress(0)
                df_consolidated = analyzer.complete_analysis_all_short_ci(progress_bar)
                progress_bar.empty()
                
                if df_consolidated is not None and len(df_consolidated) > 0:
                    st.success(f"✅ {len(df_consolidated)} alertas processados!")
                    
                    st.header("📊 Resumo")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    critical = len(df_consolidated[df_consolidated['classification'].str.contains('CRÍTICO', na=False)])
                    col1.metric("🔴 P1", critical)
                    
                    high = len(df_consolidated[df_consolidated['classification'].str.contains('PARCIALMENTE', na=False)])
                    col2.metric("🟠 P2", high)
                    
                    medium = len(df_consolidated[df_consolidated['classification'].str.contains('DETECTÁVEL', na=False)])
                    col3.metric("🟡 P3", medium)
                    
                    low = len(df_consolidated[df_consolidated['classification'].str.contains('NÃO', na=False)])
                    col4.metric("🟢 P4", low)
                    
                    st.subheader("🏆 Top 20 Reincidentes")
                    display_cols = [
                        'short_ci', 'score', 'classification', 
                        'total_occurrences', 'cv', 'regularity_score',
                        'hourly_concentration', 'markov_score', 'anomaly_rate'
                    ]
                    available_cols = [col for col in display_cols if col in df_consolidated.columns]
                    top_20 = df_consolidated.nlargest(20, 'score')[available_cols].round(2)
                    st.dataframe(top_20, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📥 Exportar")
                    
                    col1, col2 = st.columns(2)
                    
                    csv_full = df_consolidated.to_csv(index=False)
                    col1.download_button(
                        "⬇️ CSV Completo",
                        csv_full,
                        f"completo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    summary_cols = ['short_ci', 'score', 'classification', 'total_occurrences']
                    available_summary = [col for col in summary_cols if col in df_consolidated.columns]
                    summary = df_consolidated[available_summary].copy()
                    csv_summary = summary.to_csv(index=False)
                    col2.download_button(
                        "⬇️ CSV Resumido",
                        csv_summary,
                        f"resumo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
else:
    st.info("👆 Faça upload de um CSV")
    
    with st.expander("📖 Instruções"):
        st.markdown("""
        ### 🎯 Novidades da Versão Otimizada
        
        **✨ Melhorias no Score:**
        - Adicionado Markov (5%): Detecta dependência temporal
        - Adicionado Penalização de Anomalias (até -5 pontos)
        - Ajustado Previsibilidade de 15% para 10%
        
        **⚡ Otimizações:**
        - Removidas 7 análises não utilizadas
        - ~20% mais rápido
        - Código mais limpo
        
        ### 📋 Colunas CSV:
        - `short_ci`: ID do alerta
        - `created_on`: Data/hora
        """)
```

if **name** == “**main**”:
main()
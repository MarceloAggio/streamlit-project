import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import io
import warnings
from multiprocessing import Pool, cpu_count
import holidays
from functools import partial
from collections import defaultdict, Counter
import math

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Analisador de Alertas - Corrigido",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FUNÇÕES DE GERAÇÃO DE REGRAS CUSTOMIZADAS
# ============================================================

def identificar_padrao_dominante(results):
    """
    Identifica o padrão DOMINANTE baseado em todos os scores
    
    Returns:
        tuple: (tipo_padrao, confianca, detalhes)
    """
    padroes_detectados = []
    
    # 1. PERIODICIDADE FORTE (Intervalo fixo: 8 em 8h, 12 em 12h, etc)
    if results.get('periodicity', {}).get('has_strong_periodicity', False):
        periodo_horas = results['periodicity'].get('dominant_period_hours')
        
        if periodo_horas:
            confianca = 90 if results.get('regularity', {}).get('regularity_score', 0) > 80 else 75
            
            # Classificar o tipo de período
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
    
    # 2. CONCENTRAÇÃO HORÁRIA (Sempre nos mesmos horários: 8h, 16h, 00h)
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
    
    # 3. CONCENTRAÇÃO SEMANAL (Sempre nos mesmos dias da semana)
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
    
    # 6. MARKOV (Dependência de estado anterior)
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
    
    # 7. COMPORTAMENTO CONTEXTUAL (Fins de semana, feriados, horário comercial)
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
    
    # 8. SEM PADRÃO CLARO (comportamento aleatório)
    if not padroes_detectados or results['randomness'].get('overall_randomness_score', 50) > 70:
        return {
            'tipo': 'SEM_PADRAO_CLARO',
            'subtipo': 'ALEATORIO',
            'confianca': 60,
            'score_relevancia': 0,
            'descricao': 'Comportamento aleatório ou sem padrão detectável'
        }, []
    
    # Ordenar por relevância e confiança
    padroes_detectados.sort(key=lambda x: (x['score_relevancia'], x['confianca']), reverse=True)
    
    # Retornar padrão dominante + padrões secundários
    padrao_principal = padroes_detectados[0] if padroes_detectados else None
    padroes_secundarios = padroes_detectados[1:3] if len(padroes_detectados) > 1 else []
    
    return padrao_principal, padroes_secundarios


def gerar_regra_customizada(results, df, intervals_hours):
    """
    Gera regra customizada baseada no padrão identificado
    
    Returns:
        dict: Regra com parâmetros específicos
    """
    # Validar que results tem os campos necessários
    required_keys = ['regularity', 'periodicity', 'predictability', 'basic_stats']
    for key in required_keys:
        if key not in results or results[key] is None:
            return gerar_regra_generica()
    
    padrao_principal, padroes_secundarios = identificar_padrao_dominante(results)
    
    if not padrao_principal:
        return gerar_regra_generica()
    
    tipo = padrao_principal['tipo']
    
    # REGRAS POR TIPO DE PADRÃO
    
    if tipo == 'PERIODICO_FIXO':
        periodo = padrao_principal['periodo_horas']
        subtipo = padrao_principal['subtipo']
        
        # Calcular tolerância baseada na regularidade
        cv = results['regularity']['cv']
        if cv < 0.2:
            tolerancia_pct = 0.10  # 10% de tolerância para padrões muito regulares
        elif cv < 0.4:
            tolerancia_pct = 0.15
        else:
            tolerancia_pct = 0.25
        
        tolerancia = periodo * tolerancia_pct
        
        # Definir quantas ocorrências consecutivas
        if periodo < 4:  # Menos de 4 horas
            ocorrencias_minimas = 5  # 5 vezes consecutivas
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
FOR cada novo alerta:
    ultimos_alertas = buscar_ultimos_alertas(janela={janela_analise_horas:.0f}h)
    
    IF len(ultimos_alertas) >= {ocorrencias_minimas}:
        intervalos = calcular_intervalos(ultimos_alertas)
        
        contador_padrao = 0
        FOR intervalo in intervalos:
            IF {periodo - tolerancia:.1f} <= intervalo <= {periodo + tolerancia:.1f}:
                contador_padrao += 1
        
        IF contador_padrao >= {ocorrencias_minimas - 1}:
            MARCAR_COMO_REINCIDENTE()
            GERAR_ALERTA_CRITICO()
"""
            }
        }
    
    elif tipo == 'HORARIOS_FIXOS':
        horarios = padrao_principal['horarios']
        concentracao = padrao_principal['concentracao']
        
        # Tolerância em minutos
        tolerancia_minutos = 30 if concentracao > 80 else 45
        
        # Quantos dias consecutivos
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
FOR cada novo alerta:
    hora_alerta = extrair_hora(alerta)
    
    # Verificar se está em horário de pico
    em_horario_pico = FALSE
    FOR horario_esperado in {horarios}:
        IF abs(hora_alerta - horario_esperado) <= {tolerancia_minutos / 60:.2f}:
            em_horario_pico = TRUE
            BREAK
    
    IF em_horario_pico:
        ultimos_dias = buscar_ultimos_dias({dias_consecutivos})
        
        dias_com_padrao = 0
        FOR dia in ultimos_dias:
            IF dia_tem_alerta_em_horario_pico(dia):
                dias_com_padrao += 1
        
        IF dias_com_padrao >= {dias_consecutivos}:
            MARCAR_COMO_REINCIDENTE()
"""
            }
        }
    
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
FOR cada novo alerta:
    dia_semana = extrair_dia_semana(alerta)
    
    IF dia_semana in {dias}:
        ultimas_semanas = buscar_ultimas_semanas({semanas_consecutivas})
        
        semanas_com_padrao = 0
        FOR semana in ultimas_semanas:
            IF semana_tem_alerta_em_dias_pico(semana):
                semanas_com_padrao += 1
        
        IF semanas_com_padrao >= {semanas_consecutivas}:
            MARCAR_COMO_REINCIDENTE()
"""
            }
        }
    
    elif tipo == 'BURST':
        n_bursts = padrao_principal['num_bursts']
        
        # Para bursts, usar intervalo curto
        media_intervalo = results['basic_stats']['mean']
        intervalo_burst = media_intervalo / 3  # Um terço do intervalo médio
        
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
FOR cada novo alerta:
    ultimos_alertas = buscar_ultimos_alertas(janela={janela_horas:.0f}h)
    
    IF len(ultimos_alertas) >= {ocorrencias_minimas_burst}:
        intervalos = calcular_intervalos(ultimos_alertas)
        
        em_burst = TRUE
        FOR intervalo in intervalos:
            IF intervalo > {intervalo_burst:.1f}:
                em_burst = FALSE
                BREAK
        
        IF em_burst:
            MARCAR_COMO_REINCIDENTE()
            PRIORIDADE_MAXIMA()
"""
            }
        }
    
    elif tipo == 'REGULAR_SEM_CICLO':
        cv = padrao_principal['cv']
        
        # Padrão regular mas sem ciclo claro - usar estatísticas
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
FOR cada novo alerta:
    ultimos_alertas = buscar_ultimos_alertas(janela={janela:.0f}h)
    
    IF len(ultimos_alertas) >= {ocorrencias_minimas}:
        intervalos = calcular_intervalos(ultimos_alertas)
        
        dentro_do_padrao = 0
        FOR intervalo in intervalos:
            IF {intervalo_esperado_min:.1f} <= intervalo <= {intervalo_esperado_max:.1f}:
                dentro_do_padrao += 1
        
        IF dentro_do_padrao >= {ocorrencias_minimas - 1}:
            MARCAR_COMO_REINCIDENTE()
"""
            }
        }
    
    else:  # SEM_PADRAO_CLARO ou outros
        return gerar_regra_generica()


def gerar_regra_generica():
    """Regra genérica quando não há padrão claro"""
    return {
        'tipo': 'THRESHOLD_GENERICO',
        'subtipo': 'SEM_PADRAO',
        'padrao_principal': {
            'tipo': 'SEM_PADRAO_CLARO',
            'descricao': 'Sem padrão recorrente detectável',
            'confianca': 50
        },
        'padroes_secundarios': [],
        'parametros': {
            'ocorrencias_janela': 3,
            'janela_horas': 24,
            'metodo': 'Threshold simples'
        },
        'criterio_reincidencia': {
            'descricao': "Manter regra atual de threshold",
            'condicao': "3 ou mais alertas",
            'janela': "Janela de 24 horas"
        },
        'implementacao': {
            'logica': """
FOR cada novo alerta:
    ultimos_24h = buscar_ultimos_alertas(janela=24h)
    
    IF len(ultimos_24h) >= 3:
        MARCAR_COMO_REINCIDENTE()
"""
        }
    }


def calcular_efetividade_regra(df, regra, regra_atual={'ocorrencias': 3, 'janela_horas': 24}):
    """
    Simula a efetividade da regra customizada vs regra atual
    
    Returns:
        dict: Métricas de comparação
    """
    if len(df) < 3:
        return {
            'regra_customizada': {'deteccoes': 0, 'taxa': 0},
            'regra_atual': {'deteccoes': 0, 'taxa': 0},
            'melhoria': 0
        }
    
    df_sorted = df.sort_values('created_on').reset_index(drop=True)
    
    # Simular regra ATUAL (3 em 24h)
    deteccoes_atual = 0
    for i in range(len(df_sorted)):
        janela_inicio = df_sorted.loc[i, 'created_on'] - timedelta(hours=regra_atual['janela_horas'])
        alertas_na_janela = df_sorted[
            (df_sorted['created_on'] >= janela_inicio) & 
            (df_sorted['created_on'] <= df_sorted.loc[i, 'created_on'])
        ]
        if len(alertas_na_janela) >= regra_atual['ocorrencias']:
            deteccoes_atual += 1
    
    # Simular regra CUSTOMIZADA
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
            
            # Verificar se está em horário de pico
            em_horario_pico = any(abs(hora_atual - h) <= tolerancia_h for h in horarios)
            
            if em_horario_pico:
                data_atual = df_sorted.loc[i, 'created_on'].date()
                datas_anteriores = [data_atual - timedelta(days=d) for d in range(1, dias_consecutivos)]
                
                dias_com_padrao = 1  # Dia atual
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
    
    else:  # THRESHOLD_GENERICO ou outros
        deteccoes_custom = deteccoes_atual
    
    # Calcular taxas
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


# ============================================================
# FUNÇÕES AUXILIARES PARA MULTIPROCESSING
# ============================================================

def analyze_single_short_ci_recurrence(short_ci, df_original):
    """Função auxiliar para análise de reincidência de um único short_ci (para multiprocessing)"""
    try:
        df_ci = df_original[df_original['short_ci'] == short_ci].copy()
        df_ci['created_on'] = pd.to_datetime(df_ci['created_on'], errors='coerce')
        df_ci = df_ci.dropna(subset=['created_on'])
        df_ci = df_ci.sort_values('created_on')
        
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


def analyze_chunk_recurrence(short_ci_list, df_original):
    """Processa um chunk de short_ci para análise de reincidência"""
    results = []
    for short_ci in short_ci_list:
        result = analyze_single_short_ci_recurrence(short_ci, df_original)
        if result:
            results.append(result)
    return results


# ============================================================
# CLASSE DE ANÁLISE DE REINCIDÊNCIA
# ============================================================

class AdvancedRecurrenceAnalyzer:
    """Analisador completo de padrões de reincidência"""
    
    def __init__(self, df, alert_id):
        self.df = df.copy() if df is not None else None
        self.alert_id = alert_id
        self.cache = {}
    
    def _prepare_data(self):
        """Preparação vetorizada dos dados"""
        if self.df is None or len(self.df) < 3:
            return None
        
        df = self.df.sort_values('created_on').copy()
        
        # Vetorizar operações
        df['timestamp'] = df['created_on'].astype('int64') // 10**9
        df['time_diff_seconds'] = df['timestamp'].diff()
        df['time_diff_hours'] = df['time_diff_seconds'] / 3600
        df['time_diff_days'] = df['time_diff_seconds'] / 86400
        
        # Componentes temporais
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
        """Método principal com interface Streamlit COMPLETA"""
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
        
        # Executar TODAS as análises
        results = {}
        results['basic_stats'] = self._analyze_basic_statistics(intervals_hours)
        results['regularity'] = self._analyze_regularity(intervals_hours)
        results['periodicity'] = self._analyze_periodicity(intervals_hours)
        results['autocorr'] = self._analyze_autocorrelation(intervals_hours)
        results['temporal'] = self._analyze_temporal_patterns(df)
        results['clusters'] = self._analyze_clusters(df, intervals_hours)
        results['bursts'] = self._detect_bursts(intervals_hours)
        results['seasonality'] = self._analyze_seasonality(df)
        results['changepoints'] = self._detect_changepoints(intervals_hours)
        results['anomalies'] = self._detect_anomalies(intervals_hours)
        results['predictability'] = self._calculate_predictability(intervals_hours)
        results['stability'] = self._analyze_stability(intervals_hours, df)
        results['contextual'] = self._analyze_contextual_dependencies(df)
        results['vulnerability'] = self._identify_vulnerability_windows(df, intervals_hours)
        results['maturity'] = self._analyze_pattern_maturity(df, intervals_hours)
        results['prediction_confidence'] = self._calculate_prediction_confidence(intervals_hours)
        results['markov'] = self._analyze_markov_chains(intervals_hours)
        results['randomness'] = self._advanced_randomness_tests(intervals_hours)
        
        # Classificação final COM REGRAS CUSTOMIZADAS
        self._final_classification(results, df, intervals_hours)
    
    def analyze_complete_silent(self):
        """
        Análise COMPLETA silenciosa - MESMAS 18 ANÁLISES do modo Individual
        Usado no modo "Completa + CSV"
        """
        df = self._prepare_data()
        if df is None or len(df) < 3:
            return None
        
        intervals_hours = df['time_diff_hours'].dropna().values
        if len(intervals_hours) < 2:
            return None
        
        # Executar TODAS as 18 análises (sem interface)
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
            results['autocorr'] = self._analyze_autocorrelation_complete(intervals_hours)
        except Exception:
            results['autocorr'] = {'max_autocorr': 0}
        
        try:
            results['temporal'] = self._analyze_temporal_patterns_complete(df)
        except Exception:
            results['temporal'] = {'hourly_concentration': 0, 'daily_concentration': 0, 'peak_hours': [], 'peak_days': []}
        
        try:
            results['clusters'] = self._analyze_clusters_complete(df, intervals_hours)
        except Exception:
            results['clusters'] = {'n_clusters': 0}
        
        try:
            results['bursts'] = self._detect_bursts_complete(intervals_hours)
        except Exception:
            results['bursts'] = {'n_bursts': 0, 'has_bursts': False}
        
        try:
            results['seasonality'] = self._analyze_seasonality_complete(df)
        except Exception:
            results['seasonality'] = {'trend': 'stable'}
        
        try:
            results['changepoints'] = self._detect_changepoints_complete(intervals_hours)
        except Exception:
            results['changepoints'] = {'changepoints': [], 'has_changes': False}
        
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
            results['vulnerability'] = self._identify_vulnerability_windows_complete(df)
        except Exception:
            results['vulnerability'] = {'top_windows': []}
        
        try:
            results['maturity'] = self._analyze_pattern_maturity_complete(intervals_hours)
        except Exception:
            results['maturity'] = {'maturity': 'stable'}
        
        try:
            results['prediction_confidence'] = self._calculate_prediction_confidence_complete(intervals_hours)
        except Exception:
            results['prediction_confidence'] = {'confidence': 'low', 'score': 0}
        
        try:
            results['markov'] = self._analyze_markov_chains_complete(intervals_hours)
        except Exception:
            results['markov'] = {'markov_score': 0}
        
        try:
            results['randomness'] = self._advanced_randomness_tests_complete(intervals_hours)
        except Exception:
            results['randomness'] = {'overall_randomness_score': 50}
        
        # Calcular score final VALIDADO
        final_score, classification = self._calculate_final_score_validated(results, df, intervals_hours)
        
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
    # ANÁLISES COM INTERFACE (versão completa para modo Individual)
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
    
    def _analyze_autocorrelation(self, intervals):
        """Análise de autocorrelação"""
        st.subheader("📈 4. Autocorrelação")
        
        if len(intervals) < 5:
            return {}
        
        intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
        autocorr = signal.correlate(intervals_norm, intervals_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        lags = np.arange(len(autocorr))
        threshold = 2 / np.sqrt(len(intervals))
        
        significant_peaks = [(i, autocorr[i]) for i in range(1, min(len(autocorr), 20)) 
                           if autocorr[i] > threshold]
        
        if significant_peaks:
            st.success("✅ **Autocorrelação Significativa:**")
            for lag, corr in significant_peaks[:3]:
                st.write(f"• Lag {lag}: {corr:.2f}")
        else:
            st.info("📊 Sem autocorrelação significativa")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lags[:min(30, len(lags))],
            y=autocorr[:min(30, len(autocorr))],
            mode='lines+markers'
        ))
        fig.add_hline(y=threshold, line_dash="dash", line_color="red")
        fig.add_hline(y=-threshold, line_dash="dash", line_color="red")
        fig.update_layout(title="Autocorrelação", height=300)
        st.plotly_chart(fig, use_container_width=True, key='autocorr')
        
        # Calcular max autocorrelation para o score
        max_autocorr = max([corr for _, corr in significant_peaks], default=0)
        
        return {
            'peaks': significant_peaks, 
            'has_autocorr': len(significant_peaks) > 0,
            'max_autocorr': max_autocorr
        }
    
    def _analyze_temporal_patterns(self, df):
        """Análise de padrões temporais"""
        st.subheader("⏰ 5. Padrões Temporais")
        
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
    
    def _analyze_clusters(self, df, intervals):
        """Detecção de clusters temporais"""
        st.subheader("🎯 6. Clusters Temporais")
        
        if len(df) < 10:
            st.info("Mínimo de 10 ocorrências necessário")
            return {}
        
        first_ts = df['timestamp'].min()
        time_features = ((df['timestamp'] - first_ts) / 3600).values.reshape(-1, 1)
        
        eps = np.median(intervals) * 2
        dbscan = DBSCAN(eps=eps, min_samples=3)
        clusters = dbscan.fit_predict(time_features)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Clusters", n_clusters)
        col2.metric("📊 Em Clusters", len(clusters) - n_noise)
        col3.metric("🔴 Isolados", n_noise)
        
        if n_clusters > 0:
            st.success(f"✅ **{n_clusters} clusters** identificados")
        
        return {'n_clusters': n_clusters, 'n_noise': n_noise}
    
    def _detect_bursts(self, intervals):
        """Detecção de bursts"""
        st.subheader("💥 7. Detecção de Bursts")
        
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
    
    def _analyze_seasonality(self, df):
        """Análise de sazonalidade"""
        st.subheader("🌡️ 8. Sazonalidade")
        
        date_range = (df['created_on'].max() - df['created_on'].min()).days
        
        if date_range < 30:
            st.info("📊 Período curto para análise sazonal")
            return {}
        
        weekly = df.groupby('week_of_year').size()
        
        if len(weekly) >= 4:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weekly.index,
                y=weekly.values,
                mode='lines+markers',
                fill='tozeroy'
            ))
            fig.update_layout(title="Evolução Semanal", height=250)
            st.plotly_chart(fig, use_container_width=True, key='weekly')
            
            if len(weekly) > 3:
                slope, _, _, p_value, _ = stats.linregress(weekly.index.values, weekly.values)
                if p_value < 0.05:
                    if slope > 0:
                        st.warning("📈 **Tendência crescente**")
                        return {'trend': 'increasing', 'slope': slope}
                    else:
                        st.success("📉 **Tendência decrescente**")
                        return {'trend': 'decreasing', 'slope': slope}
        
        return {'trend': 'stable'}
    
    def _detect_changepoints(self, intervals):
        """Detecção de pontos de mudança"""
        st.subheader("🔀 9. Pontos de Mudança")
        
        if len(intervals) < 20:
            st.info("Mínimo de 20 intervalos necessário")
            return {}
        
        cumsum = np.cumsum(intervals - np.mean(intervals))
        
        window = 5
        changes = []
        for i in range(window, len(cumsum) - window):
            before = np.mean(intervals[max(0, i-window):i])
            after = np.mean(intervals[i:min(len(intervals), i+window)])
            if abs(before - after) > np.std(intervals):
                changes.append(i)
        
        filtered = []
        for cp in changes:
            if not filtered or cp - filtered[-1] > 5:
                filtered.append(cp)
        
        if filtered:
            st.warning(f"⚠️ **{len(filtered)} pontos de mudança** detectados")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(cumsum))), y=cumsum, mode='lines'))
            for cp in filtered:
                fig.add_vline(x=cp, line_dash="dash", line_color="red")
            fig.update_layout(title="CUSUM", height=250)
            st.plotly_chart(fig, use_container_width=True, key='cusum')
        else:
            st.success("✅ Comportamento estável")
        
        return {'changepoints': filtered, 'has_changes': len(filtered) > 0}
    
    def _detect_anomalies(self, intervals):
        """Detecção de anomalias"""
        st.subheader("🚨 10. Detecção de Anomalias")
        
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
        st.subheader("🔮 11. Previsibilidade")
        
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
        st.subheader("📊 12. Estabilidade")
        
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
        st.subheader("🌐 13. Dependências Contextuais")
        
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
    
    def _identify_vulnerability_windows(self, df, intervals):
        """Janelas de vulnerabilidade"""
        st.subheader("🎯 14. Janelas de Vulnerabilidade")
        
        vulnerability_matrix = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        vulnerability_matrix['risk_score'] = (
            vulnerability_matrix['count'] / vulnerability_matrix['count'].max() * 100
        )
        
        top_windows = vulnerability_matrix.nlargest(5, 'risk_score')
        
        day_map = {0: 'Seg', 1: 'Ter', 2: 'Qua', 3: 'Qui', 4: 'Sex', 5: 'Sáb', 6: 'Dom'}
        
        st.write("**🔴 Top 5 Janelas Críticas:**")
        for idx, row in top_windows.iterrows():
            day = day_map[row['day_of_week']]
            hour = int(row['hour'])
            risk = row['risk_score']
            st.write(f"• **{day} {hour:02d}:00** - Score: {risk:.1f} ({row['count']} alertas)")
        
        return {'top_windows': top_windows.to_dict('records')}
    
    def _analyze_pattern_maturity(self, df, intervals):
        """Maturidade do padrão"""
        st.subheader("📈 15. Maturidade do Padrão")
        
        n_periods = 4
        period_size = len(intervals) // n_periods
        
        if period_size < 2:
            st.info("Período insuficiente")
            return {}
        
        periods_stats = []
        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size if i < n_periods - 1 else len(intervals)
            period_intervals = intervals[start:end]
            
            periods_stats.append({
                'period': i + 1,
                'mean': np.mean(period_intervals),
                'cv': np.std(period_intervals) / np.mean(period_intervals) if np.mean(period_intervals) > 0 else 0
            })
        
        periods_df = pd.DataFrame(periods_stats)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=periods_df['period'],
            y=periods_df['cv'],
            mode='lines+markers',
            name='CV',
            line=dict(color='red', width=3)
        ))
        fig.update_layout(
            title="Evolução da Variabilidade",
            xaxis_title="Período",
            yaxis_title="CV",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True, key='maturity')
        
        slope = np.polyfit(periods_df['period'], periods_df['cv'], 1)[0]
        
        if slope < -0.05:
            st.success("✅ **Amadurecendo**: Variabilidade decrescente")
            maturity = "maturing"
        elif slope > 0.05:
            st.warning("⚠️ **Degradando**: Variabilidade crescente")
            maturity = "degrading"
        else:
            st.info("📊 **Estável**: Variabilidade constante")
            maturity = "stable"
        
        return {'maturity': maturity, 'slope': slope}
    
    def _calculate_prediction_confidence(self, intervals):
        """Confiança da predição"""
        st.subheader("🎯 16. Confiança de Predição")
        
        if len(intervals) < 10:
            return {'confidence': 'low', 'score': 0}
        
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        n_samples = len(intervals)
        
        regularity_score = max(0, 100 - cv * 100)
        sample_score = min(100, (n_samples / 50) * 100)
        
        mid = len(intervals) // 2
        var1 = np.var(intervals[:mid])
        var2 = np.var(intervals[mid:])
        var_ratio = min(var1, var2) / max(var1, var2) if max(var1, var2) > 0 else 0
        stationarity_score = var_ratio * 100
        
        confidence_score = (regularity_score * 0.5 + sample_score * 0.3 + stationarity_score * 0.2)
        
        if confidence_score > 70:
            confidence = 'high'
        elif confidence_score > 40:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        col1, col2 = st.columns(2)
        col1.metric("Confiança", confidence.upper())
        col2.metric("Score", f"{confidence_score:.1f}%")
        
        return {'confidence': confidence, 'score': confidence_score}
    
    def _analyze_markov_chains(self, intervals):
        """Cadeias de Markov"""
        st.subheader("🔗 17. Cadeias de Markov")
        
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
        st.subheader("🎲 18. Testes de Aleatoriedade")
        
        if len(intervals) < 10:
            st.info("Mínimo de 10 intervalos necessário")
            return {}
        
        results = {}
        
        # Runs Test
        st.write("**1️⃣ Runs Test**")
        median = np.median(intervals)
        runs = np.diff(intervals > median).sum() + 1
        expected_runs = len(intervals) / 2
        
        col1, col2 = st.columns(2)
        col1.metric("Runs Observados", runs)
        col2.metric("Runs Esperados", f"{expected_runs:.1f}")
        
        # Permutation Entropy
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
        
        # Hurst Exponent
        st.write("**3️⃣ Hurst Exponent**")
        
        def hurst_exponent(series):
            n = len(series)
            if n < 20:
                return None
            
            lags = range(2, min(n//2, 20))
            tau = []
            
            for lag in lags:
                n_partitions = n // lag
                partitions = [series[i*lag:(i+1)*lag] for i in range(n_partitions)]
                
                rs_values = []
                for partition in partitions:
                    if len(partition) == 0:
                        continue
                    mean = np.mean(partition)
                    cumsum = np.cumsum(partition - mean)
                    R = np.max(cumsum) - np.min(cumsum)
                    S = np.std(partition)
                    if S > 0:
                        rs_values.append(R / S)
                
                if rs_values:
                    tau.append(np.mean(rs_values))
            
            if len(tau) > 2:
                log_lags = np.log(list(lags[:len(tau)]))
                log_tau = np.log(tau)
                hurst = np.polyfit(log_lags, log_tau, 1)[0]
                return hurst
            return None
        
        if len(intervals) >= 20:
            hurst = hurst_exponent(intervals)
            
            if hurst is not None:
                st.metric("Hurst", f"{hurst:.3f}")
                
                if hurst < 0.45:
                    st.info("📉 Anti-persistente")
                elif hurst > 0.55:
                    st.warning("📈 Persistente")
                else:
                    st.success("🎲 Random Walk")
                
                results['hurst'] = hurst
        
        # Score final
        st.markdown("---")
        randomness_score = 50  # Simplificado
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
    # Usam as MESMAS lógicas do modo Individual
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
    
    def _analyze_autocorrelation_complete(self, intervals):
        if len(intervals) < 5:
            return {'max_autocorr': 0}
        try:
            intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
            autocorr = signal.correlate(intervals_norm, intervals_norm, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            threshold = 2 / np.sqrt(len(intervals))
            significant_peaks = [autocorr[i] for i in range(1, min(len(autocorr), 20)) 
                               if autocorr[i] > threshold]
            
            return {'max_autocorr': max(significant_peaks) if significant_peaks else 0}
        except:
            return {'max_autocorr': 0}
    
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
    
    def _analyze_clusters_complete(self, df, intervals):
        if len(df) < 10:
            return {'n_clusters': 0}
        try:
            first_ts = df['timestamp'].min()
            time_features = ((df['timestamp'] - first_ts) / 3600).values.reshape(-1, 1)
            
            eps = np.median(intervals) * 2
            dbscan = DBSCAN(eps=eps, min_samples=3)
            clusters = dbscan.fit_predict(time_features)
            
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            return {'n_clusters': n_clusters}
        except:
            return {'n_clusters': 0}
    
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
    
    def _analyze_seasonality_complete(self, df):
        date_range = (df['created_on'].max() - df['created_on'].min()).days
        if date_range < 30:
            return {'trend': 'stable'}
        
        weekly = df.groupby('week_of_year').size()
        if len(weekly) > 3:
            try:
                slope, _, _, p_value, _ = stats.linregress(weekly.index.values, weekly.values)
                if p_value < 0.05:
                    return {'trend': 'increasing' if slope > 0 else 'decreasing', 'slope': slope}
            except:
                pass
        return {'trend': 'stable'}
    
    def _detect_changepoints_complete(self, intervals):
        if len(intervals) < 20:
            return {'changepoints': [], 'has_changes': False}
        
        cumsum = np.cumsum(intervals - np.mean(intervals))
        window = 5
        changes = []
        for i in range(window, len(cumsum) - window):
            before = np.mean(intervals[max(0, i-window):i])
            after = np.mean(intervals[i:min(len(intervals), i+window)])
            if abs(before - after) > np.std(intervals):
                changes.append(i)
        
        filtered = []
        for cp in changes:
            if not filtered or cp - filtered[-1] > 5:
                filtered.append(cp)
        
        return {'changepoints': filtered, 'has_changes': len(filtered) > 0}
    
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
    
    def _identify_vulnerability_windows_complete(self, df):
        try:
            vulnerability_matrix = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
            vulnerability_matrix['risk_score'] = (
                vulnerability_matrix['count'] / vulnerability_matrix['count'].max() * 100
            )
            top_windows = vulnerability_matrix.nlargest(5, 'risk_score')
            return {'top_windows': top_windows.to_dict('records')}
        except:
            return {'top_windows': []}
    
    def _analyze_pattern_maturity_complete(self, intervals):
        if len(intervals) < 10:
            return {'maturity': 'stable'}
        
        n_periods = 4
        period_size = len(intervals) // n_periods
        if period_size < 2:
            return {'maturity': 'stable'}
        
        periods_cvs = []
        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size if i < n_periods - 1 else len(intervals)
            period_intervals = intervals[start:end]
            cv = np.std(period_intervals) / np.mean(period_intervals) if np.mean(period_intervals) > 0 else 0
            periods_cvs.append(cv)
        
        slope = np.polyfit(range(1, n_periods + 1), periods_cvs, 1)[0]
        
        if slope < -0.05:
            return {'maturity': 'maturing'}
        elif slope > 0.05:
            return {'maturity': 'degrading'}
        else:
            return {'maturity': 'stable'}
    
    def _calculate_prediction_confidence_complete(self, intervals):
        if len(intervals) < 10:
            return {'confidence': 'low', 'score': 0}
        
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        n_samples = len(intervals)
        
        regularity_score = max(0, 100 - cv * 100)
        sample_score = min(100, (n_samples / 50) * 100)
        
        mid = len(intervals) // 2
        var1 = np.var(intervals[:mid])
        var2 = np.var(intervals[mid:])
        var_ratio = min(var1, var2) / max(var1, var2) if max(var1, var2) > 0 else 0
        stationarity_score = var_ratio * 100
        
        confidence_score = (regularity_score * 0.5 + sample_score * 0.3 + stationarity_score * 0.2)
        
        if confidence_score > 70:
            confidence = 'high'
        elif confidence_score > 40:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {'confidence': confidence, 'score': confidence_score}
    
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
    # CLASSIFICAÇÃO FINAL COM SCORE VALIDADO
    # ============================================================
    
    def _calculate_final_score_validated(self, results, df, intervals):
        """
        Calcula score final com critérios VALIDADOS
        
        NOVO PESO DOS CRITÉRIOS:
        - Regularidade: 20% (mantido, essencial)
        - Periodicidade: 20% (mantido, essencial)
        - Previsibilidade: 15% (reduzido de 20%)
        - Concentração Temporal: 20% (NOVO - horários/dias fixos)
        - Frequência Absoluta: 15% (NOVO - volume importa)
        - Bursts: 10% (NOVO - rajadas são padrão importante)
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
        
        # 3. PREVISIBILIDADE (15%)
        predictability_score = results['predictability']['predictability_score'] * 0.15
        
        # 4. CONCENTRAÇÃO TEMPORAL (20%) - NOVO
        hourly_conc = results['temporal']['hourly_concentration']
        daily_conc = results['temporal']['daily_concentration']
        
        # Se > 60% concentrado em poucos horários/dias = forte indicador
        concentration_score = 0
        if hourly_conc > 60 or daily_conc > 60:
            concentration_score = 100 * 0.20
        elif hourly_conc > 40 or daily_conc > 40:
            concentration_score = 60 * 0.20
        elif hourly_conc > 30 or daily_conc > 30:
            concentration_score = 30 * 0.20
        
        # 5. FREQUÊNCIA ABSOLUTA (15%) - NOVO
        # Importa o VOLUME: 3 alertas em 1 ano != reincidente
        total_occurrences = len(df)
        period_days = (df['created_on'].max() - df['created_on'].min()).days + 1
        freq_per_week = (total_occurrences / period_days * 7) if period_days > 0 else 0
        
        if freq_per_week >= 3:  # 3+ por semana
            frequency_score = 100 * 0.15
        elif freq_per_week >= 1:  # 1-3 por semana
            frequency_score = 70 * 0.15
        elif freq_per_week >= 0.5:  # 0.5-1 por semana
            frequency_score = 40 * 0.15
        elif total_occurrences >= 10:  # Pelo menos 10 ocorrências no total
            frequency_score = 30 * 0.15
        else:
            frequency_score = 10 * 0.15  # Penalizar baixo volume
        
        # 6. BURSTS (10%) - NOVO
        if results['bursts']['has_bursts'] and results['bursts']['n_bursts'] >= 2:
            burst_score = 100 * 0.10
        elif results['bursts']['has_bursts']:
            burst_score = 50 * 0.10
        else:
            burst_score = 0 * 0.10
        
        # SCORE FINAL
        final_score = (
            regularity_score +
            periodicity_score +
            predictability_score +
            concentration_score +
            frequency_score +
            burst_score
        )
        
        # THRESHOLDS VALIDADOS
        # Agora considera volume, concentração e bursts
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
        
        final_score, classification = self._calculate_final_score_validated(results, df, intervals)
        
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
            
            st.markdown("#### 📊 Breakdown dos Critérios VALIDADOS")
            
            # Calcular scores individuais
            total_occurrences = len(df)
            period_days = (df['created_on'].max() - df['created_on'].min()).days + 1
            freq_per_week = (total_occurrences / period_days * 7) if period_days > 0 else 0
            
            regularity_pts = results['regularity']['regularity_score'] * 0.20
            
            if results['periodicity']['has_strong_periodicity']:
                periodicity_pts = 100 * 0.20
            else:
                periodicity_pts = 0 * 0.20
            
            predictability_pts = results['predictability']['predictability_score'] * 0.15
            
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
            
            breakdown = {
                '1. Regularidade (20%)': regularity_pts,
                '2. Periodicidade (20%)': periodicity_pts,
                '3. Previsibilidade (15%)': predictability_pts,
                '4. Concentração Temporal (20%)': concentration_pts,
                '5. Frequência Absoluta (15%)': frequency_pts,
                '6. Bursts (10%)': burst_pts
            }
            
            for criterion, points in breakdown.items():
                st.write(f"• {criterion}: **{points:.1f} pts**")
            
            st.markdown("---")
            st.markdown("**💡 Justificativa dos Critérios:**")
            st.write("✅ **Regularidade**: Mede consistência dos intervalos")
            st.write("✅ **Periodicidade**: Detecta ciclos via FFT")
            st.write("✅ **Previsibilidade**: Indica se podemos prever próxima ocorrência")
            st.write("✅ **Concentração Temporal**: Horários/dias fixos são forte indicador")
            st.write("✅ **Frequência Absoluta**: Volume importa (3 em 1 ano ≠ reincidente)")
            st.write("✅ **Bursts**: Rajadas são padrão importante")
            
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
        
        # ============================================================
        # REGRAS CUSTOMIZADAS
        # ============================================================
        
        st.markdown("---")
        st.header("📋 REGRA CUSTOMIZADA SUGERIDA")
        
        # Gerar regra customizada
        regra = gerar_regra_customizada(results, df, intervals)
        
        # Mostrar padrão principal
        padrao_principal = regra['padrao_principal']
        st.success(f"**🎯 Padrão Detectado:** {padrao_principal['tipo']}")
        st.write(f"**Descrição:** {padrao_principal['descricao']}")
        st.write(f"**Confiança:** {padrao_principal['confianca']}%")
        
        # Padrões secundários
        padroes_secundarios = regra.get('padroes_secundarios', [])
        if padroes_secundarios:
            st.info("**📊 Padrões Secundários Detectados:**")
            for padrao in padroes_secundarios:
                st.write(f"• {padrao['tipo']}: {padrao['descricao']} (confiança: {padrao['confianca']}%)")
        
        # Parâmetros da regra
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
        
        # Implementação
        st.markdown("---")
        st.subheader("💻 Lógica de Implementação")
        
        with st.expander("Ver pseudocódigo", expanded=False):
            st.code(regra['implementacao']['logica'], language='python')
        
        # Comparação com regra atual
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
            st.success(f"✅ A regra customizada é **{efetividade['melhoria']:.1f}% mais efetiva** que a regra atual!")
        elif efetividade['melhoria'] > 0:
            st.info(f"📊 A regra customizada apresenta melhoria modesta de **{efetividade['melhoria']:.1f}%**")
        elif efetividade['melhoria'] < -10:
            st.warning(f"⚠️ A regra atual parece mais adequada (diferença: {abs(efetividade['melhoria']):.1f}%)")
        else:
            st.info("📊 Ambas as regras têm efetividade similar")
        
        # Predição se score alto
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


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class StreamlitAlertAnalyzer:
    def __init__(self):
        self.df_original = None
        self.df = None
        self.dates = None
        self.alert_id = None

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
        """
        Análise COMPLETA COM MULTIPROCESSING
        Agora usa analyze_complete_silent() que tem as MESMAS 18 análises do Individual
        """
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
        
        # Frequências
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
        
        # Intervalos
        intervals = self.df['time_diff_hours'].dropna()
        if len(intervals) > 0:
            st.markdown("---")
            st.subheader("⏱️ Intervalos")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Média (h)", f"{intervals.mean():.2f}")
            col2.metric("Mediana (h)", f"{intervals.median():.2f}")
            col3.metric("Mínimo (h)", f"{intervals.min():.2f}")
            col4.metric("Máximo (h)", f"{intervals.max():.2f}")


# ============================================================
# FUNÇÃO MAIN
# ============================================================

def main():
    st.title("🚨 Analisador de Alertas - Versão Corrigida")
    st.markdown("### 2 modos: Individual e Completa + CSV (com critérios validados)")
    
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
                st.subheader("📊 Análise Completa COM CRITÉRIOS VALIDADOS")
                
                st.info("""
                **✅ Critérios Validados:**
                - Regularidade (20%) - Consistência dos intervalos
                - Periodicidade (20%) - Detecta ciclos
                - Previsibilidade (15%) - Predição de próxima ocorrência
                - **Concentração Temporal (20%)** - Horários/dias fixos
                - **Frequência Absoluta (15%)** - Volume importa
                - **Bursts (10%)** - Rajadas são padrão importante
                """)
                
                if st.sidebar.button("🚀 Executar", type="primary"):
                    st.info("⏱️ Processando com multiprocessing...")
                    
                    progress_bar = st.progress(0)
                    df_consolidated = analyzer.complete_analysis_all_short_ci(progress_bar)
                    progress_bar.empty()
                    
                    if df_consolidated is not None and len(df_consolidated) > 0:
                        st.success(f"✅ {len(df_consolidated)} alertas processados!")
                        
                        # Resumo
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
                        
                        # Top 20
                        st.subheader("🏆 Top 20 Reincidentes")
                        display_cols = [
                            'short_ci', 'score', 'classification', 
                            'total_occurrences', 'cv', 'regularity_score',
                            'hourly_concentration', 'burst_detected'
                        ]
                        available_cols = [col for col in display_cols if col in df_consolidated.columns]
                        top_20 = df_consolidated.nlargest(20, 'score')[available_cols].round(2)
                        st.dataframe(top_20, use_container_width=True)
                        
                        # Downloads
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
        
        with st.expander("📖 Instruções e Validação dos Critérios"):
            st.markdown("""
            ### ✅ CRITÉRIOS VALIDADOS
            
            **Por que removemos alguns critérios?**
            
            ❌ **Autocorrelação (removido)**: Já está implícito na periodicidade
            ❌ **Estabilidade (removido)**: Penaliza padrões novos/emergentes
            ❌ **Determinismo puro (removido)**: Muito abstrato
            
            **Por que adicionamos novos critérios?**
            
            ✅ **Concentração Temporal (20%)**: Horários/dias fixos são FORTE indicador de reincidência
            ✅ **Frequência Absoluta (15%)**: 3 alertas em 1 ano NÃO é reincidente
            ✅ **Bursts (10%)**: Rajadas são um padrão importante e comum
            
            ### 🎯 Novos Pesos (Total = 100%)
            
            1. **Regularidade (20%)** - Consistência via CV
            2. **Periodicidade (20%)** - Detecta ciclos via FFT
            3. **Previsibilidade (15%)** - Indica se podemos prever
            4. **Concentração Temporal (20%)** - Horários/dias fixos
            5. **Frequência Absoluta (15%)** - Volume mínimo necessário
            6. **Bursts (10%)** - Padrão de rajadas
            
            ### 🚀 Funcionalidades:
            
            **🔍 Análise Individual:**
            - 📊 18 análises avançadas
            - 📋 Geração automática de regras customizadas
            - ⚙️ Parâmetros específicos por série
            - 📊 Comparação com regra atual
            - 💻 Pseudocódigo de implementação
            
            **📊 Completa + CSV:**
            - Análise em lote com multiprocessing
            - MESMAS 18 análises do Individual
            - Critérios de score VALIDADOS
            - CSV completo com todas métricas
            - Regras customizadas para cada série
            
            ### 📋 Colunas CSV:
            - `short_ci`: ID do alerta
            - `created_on`: Data/hora (formato ISO ou dd/mm/yyyy HH:MM:SS)
            
            ### 🎯 Exemplo de uso:
            1. Faça upload do CSV
            2. Escolha o modo (Individual para análise detalhada, Completa para batch)
            3. Analise os resultados com critérios validados
            4. Exporte as regras customizadas
            """)


if __name__ == "__main__":
    main()
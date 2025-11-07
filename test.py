import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import warnings
from multiprocessing import Pool, cpu_count
import holidays
from functools import partial
from collections import Counter
import math

warnings.filterwarnings('ignore')

st.set_page_config(
  page_title="Analisador de Alertas",
  page_icon="🚨",
  layout="wide",
  initial_sidebar_state="expanded"
)


# ----------------------------
# Helpers para multiprocessing
# ----------------------------
def analyze_single_u_alert_id_recurrence(u_alert_id, df_original):
  try:
    df_ci = df_original[df_original['u_alert_id'] == u_alert_id].copy()
    df_ci['created_on'] = pd.to_datetime(df_ci['created_on'], errors='coerce')
    df_ci = df_ci.dropna(subset=['created_on']).sort_values('created_on')

    if len(df_ci) < 3:
      return {
        'u_alert_id': u_alert_id,
        'total_occurrences': len(df_ci),
        'score': 0,
        'classification': '⚪ DADOS INSUFICIENTES',
        'mean_interval_hours': None,
        'cv': None,
        'cv_robusto': None,
        'cv_metodo': None,
        'regularity_score': 0,
        'periodicity_detected': False,
        'predictability_score': 0,
        'outliers_count': 0,
        'outliers_percent': 0
      }

    analyzer = AdvancedRecurrenceAnalyzer(df_ci, u_alert_id)
    return analyzer.analyze_complete_silent()

  except Exception as e:
    return {
      'u_alert_id': u_alert_id,
      'total_occurrences': 0,
      'score': 0,
      'classification': f'⚪ ERRO: {str(e)[:50]}',
      'mean_interval_hours': None,
      'cv': None,
      'cv_robusto': None,
      'cv_metodo': None,
      'regularity_score': 0,
      'periodicity_detected': False,
      'predictability_score': 0,
      'outliers_count': 0,
      'outliers_percent': 0
    }


def analyze_chunk_recurrence(u_alert_id_list, df_original):
  results = []
  for u_alert_id in u_alert_id_list:
    result = analyze_single_u_alert_id_recurrence(u_alert_id, df_original)
    if result:
      results.append(result)
  return results


# ============================================================
# AdvancedRecurrenceAnalyzer: análise (UI render opcional)
# ============================================================
class AdvancedRecurrenceAnalyzer:
  def __init__(self, df, alert_id):
    self.df = df.copy() if df is not None else None
    self.alert_id = alert_id

  # ========== MÉTODOS DE DETECÇÃO DE OUTLIERS ==========
  def _detect_outliers_comprehensive(self, intervals):
    """
    Detecta outliers usando múltiplos métodos
    Retorna índices e informações sobre os outliers
    """
    results = {
      'outliers_indices': set(),
      'outliers_values': [],
      'methods': {}
    }
    
    if len(intervals) < 3:
      return results
    
    # Método 1: Z-Score (>3)
    z_scores = np.abs(stats.zscore(intervals))
    z_outliers = np.where(z_scores > 3)[0]
    results['methods']['z_score'] = {
      'count': len(z_outliers),
      'indices': z_outliers.tolist(),
      'threshold': 3
    }
    results['outliers_indices'].update(z_outliers)
    
    # Método 2: IQR
    q1, q3 = np.percentile(intervals, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_outliers = np.where((intervals < lower) | (intervals > upper))[0]
    results['methods']['iqr'] = {
      'count': len(iqr_outliers),
      'indices': iqr_outliers.tolist(),
      'lower': lower,
      'upper': upper
    }
    results['outliers_indices'].update(iqr_outliers)
    
    # Método 3: Isolation Forest (se houver dados suficientes)
    if len(intervals) >= 10:
      iso_forest = IsolationForest(contamination=0.1, random_state=42)
      predictions = iso_forest.fit_predict(intervals.reshape(-1, 1))
      iso_outliers = np.where(predictions == -1)[0]
      results['methods']['isolation_forest'] = {
        'count': len(iso_outliers),
        'indices': iso_outliers.tolist()
      }
      results['outliers_indices'].update(iso_outliers)
    
    # Consolidar resultados
    results['outliers_indices'] = sorted(list(results['outliers_indices']))
    results['outliers_values'] = intervals[results['outliers_indices']].tolist()
    results['total_outliers'] = len(results['outliers_indices'])
    results['outliers_percent'] = (results['total_outliers'] / len(intervals)) * 100
    
    return results

  # ========== MÉTODOS DE CV ROBUSTO ==========
  def _cv_robusto(self, intervals):
    """CV baseado em mediana e MAD"""
    mediana = np.median(intervals)
    mad = np.median(np.abs(intervals - mediana))
    cv_robust = mad / mediana if mediana > 0 else float('inf')
    return float(cv_robust)

  def _cv_classico(self, intervals):
    """CV clássico"""
    return float(np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf'))

  def _cv_winsorizado(self, intervals, limits=[0.1, 0.1]):
    """CV Winsorizado"""
    from scipy.stats import mstats
    intervalos_wins = mstats.winsorize(intervals, limits=limits)
    cv = np.std(intervalos_wins) / np.mean(intervalos_wins) if np.mean(intervalos_wins) > 0 else float('inf')
    return float(cv)

  def _cv_iqr(self, intervals):
    """CV baseado em IQR"""
    q1 = np.percentile(intervals, 25)
    q3 = np.percentile(intervals, 75)
    mediana = np.median(intervals)
    iqr = q3 - q1
    cv_iqr = iqr / mediana if mediana > 0 else float('inf')
    return float(cv_iqr)

  def _cv_adaptativo(self, intervals):
    """
    Escolhe automaticamente o melhor CV baseado nos dados
    THRESHOLDS AJUSTADOS para ser menos conservador
    """
    if len(intervals) < 2:
      return {
        'cv': float('inf'),
        'metodo': 'insuficiente',
        'cv_classico': float('inf'),
        'cv_robusto': float('inf'),
        'cv_winsorizado': float('inf'),
        'cv_iqr': float('inf'),
        'outliers_percent': 0,
        'diferenca_percentual': 0
      }

    # Calcular todos os CVs
    cv_classico = self._cv_classico(intervals)
    cv_robusto = self._cv_robusto(intervals)
    cv_wins = self._cv_winsorizado(intervals)
    cv_iqr_val = self._cv_iqr(intervals)
    
    # Detectar outliers com Z-score
    z_scores = np.abs(stats.zscore(intervals))
    outliers_percent = np.sum(z_scores > 3) / len(intervals)
    
    # Calcular diferença entre clássico e robusto
    diferenca_percentual = abs(cv_classico - cv_robusto) / cv_robusto * 100 if cv_robusto > 0 else 0
    
    # LÓGICA AJUSTADA - MENOS CONSERVADORA
    if outliers_percent > 0.20:  # Mudei de 15% para 20%
      cv_usado = cv_robusto
      metodo = 'Robusto (MAD)'
    elif outliers_percent > 0.15:  # Mudei de 10% para 15%
      cv_usado = cv_robusto
      metodo = 'Robusto (MAD)'
    elif diferenca_percentual > 30:  # Mudei de 20% para 30%
      cv_usado = cv_robusto
      metodo = 'Robusto (MAD)'
    elif outliers_percent > 0.08:  # Mudei de 5% para 8%
      cv_usado = cv_wins
      metodo = 'Winsorizado'
    else:  # Dados bem comportados
      cv_usado = cv_classico
      metodo = 'Clássico'
    
    return {
      'cv': cv_usado,
      'metodo': metodo,
      'cv_classico': cv_classico,
      'cv_robusto': cv_robusto,
      'cv_winsorizado': cv_wins,
      'cv_iqr': cv_iqr_val,
      'outliers_percent': outliers_percent * 100,
      'diferenca_percentual': diferenca_percentual
    }

  # ========== FIM DOS MÉTODOS DE CV ==========

  def _prepare_data(self):
    if self.df is None or len(self.df) < 3:
      return None
    df = self.df.sort_values('created_on').copy()
    df['created_on'] = pd.to_datetime(df['created_on'], errors='coerce')
    df = df.dropna(subset=['created_on'])
    df['timestamp'] = df['created_on'].astype('int64') // 10**9
    df['time_diff_seconds'] = df['timestamp'].diff()
    df['time_diff_hours'] = df['time_diff_seconds'] / 3600
    dt = df['created_on'].dt
    df['hour'] = dt.hour
    df['day_of_week'] = dt.dayofweek
    df['day_of_month'] = dt.day
    df['week_of_year'] = dt.isocalendar().week
    df['month'] = dt.month
    df['day_name'] = dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_business_hours'] = (df['hour'] >= 9) & (df['hour'] <= 17)
    return df

  # Main public methods
  def analyze(self):
    """Modo interativo (Streamlit)"""
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

    results = {}
    
    # NOVA: Análise de Outliers PRIMEIRO
    results['outliers'] = self._analyze_outliers_detailed(intervals_hours, render=True)
    
    # Demais análises
    results['basic_stats'] = self._analyze_basic_statistics(intervals_hours, render=True)
    results['regularity'] = self._analyze_regularity(intervals_hours, render=True)
    results['periodicity'] = self._analyze_periodicity(intervals_hours, render=True)
    results['autocorr'] = self._analyze_autocorrelation(intervals_hours, render=True)
    results['temporal'] = self._analyze_temporal_patterns(df, render=True)
    results['clusters'] = self._analyze_clusters(df, intervals_hours, render=True)
    results['bursts'] = self._detect_bursts(intervals_hours, render=True)
    results['seasonality'] = self._analyze_seasonality(df, render=True)
    results['changepoints'] = self._detect_changepoints(intervals_hours, render=True)
    results['anomalies'] = self._detect_anomalies(intervals_hours, render=True)
    results['predictability'] = self._calculate_predictability(intervals_hours, render=True)
    results['stability'] = self._analyze_stability(intervals_hours, df, render=True)
    results['contextual'] = self._analyze_contextual_dependencies(df, render=True)
    results['vulnerability'] = self._identify_vulnerability_windows(df, intervals_hours, render=True)
    results['maturity'] = self._analyze_pattern_maturity(df, intervals_hours, render=True)
    results['prediction_confidence'] = self._calculate_prediction_confidence(intervals_hours, render=True)
    results['markov'] = self._analyze_markov_chains(intervals_hours, render=True)
    results['randomness'] = self._advanced_randomness_tests(intervals_hours, render=True)

    self._final_classification(results, df, intervals_hours)

  def analyze_complete_silent(self):
    """Modo silencioso para batch"""
    df = self._prepare_data()
    if df is None or len(df) < 3:
      return None
    intervals_hours = df['time_diff_hours'].dropna().values
    if len(intervals_hours) < 2:
      return None

    results = {}
    
    # Análise de outliers
    try:
      outliers_info = self._detect_outliers_comprehensive(intervals_hours)
      results['outliers'] = outliers_info
    except Exception:
      results['outliers'] = {'total_outliers': 0, 'outliers_percent': 0}
    
    # Demais análises
    try:
      results['basic_stats'] = self._analyze_basic_statistics(intervals_hours, render=False)
    except Exception:
      results['basic_stats'] = {'mean': 0, 'median': 0, 'std': 0, 'cv': 0, 'cv_robusto': 0, 'cv_metodo': 'erro'}

    try:
      results['regularity'] = self._analyze_regularity(intervals_hours, render=False)
    except Exception:
      results['regularity'] = {'cv': 0, 'regularity_score': 0}

    try:
      results['periodicity'] = self._analyze_periodicity(intervals_hours, render=False)
    except Exception:
      results['periodicity'] = {'has_strong_periodicity': False, 'has_moderate_periodicity': False, 'dominant_period_hours': None}

    try:
      results['predictability'] = self._calculate_predictability(intervals_hours, render=False)
    except Exception:
      results['predictability'] = {'predictability_score': 0, 'next_expected_hours': 0}

    try:
      results['temporal'] = self._analyze_temporal_patterns(df, render=False)
    except Exception:
      results['temporal'] = {'hourly_concentration': 0, 'daily_concentration': 0, 'peak_hours': [], 'peak_days': []}

    # calcular score final
    final_score, classification = self._calculate_final_score_validated(results, df, intervals_hours)

    return {
      'u_alert_id': self.alert_id,
      'total_occurrences': len(df),
      'score': final_score,
      'classification': classification,
      'mean_interval_hours': results['basic_stats'].get('mean'),
      'median_interval_hours': results['basic_stats'].get('median'),
      'cv': results['basic_stats'].get('cv'),
      'cv_classico': results['basic_stats'].get('cv_classico'),
      'cv_robusto': results['basic_stats'].get('cv_robusto'),
      'cv_winsorizado': results['basic_stats'].get('cv_winsorizado'),
      'cv_iqr': results['basic_stats'].get('cv_iqr'),
      'cv_metodo': results['basic_stats'].get('cv_metodo'),
      'outliers_count': results['outliers'].get('total_outliers', 0),
      'outliers_percent': results['outliers'].get('outliers_percent', 0),
      'diferenca_cv_percent': results['basic_stats'].get('diferenca_percentual'),
      'regularity_score': results['regularity'].get('regularity_score'),
      'periodicity_detected': results['periodicity'].get('has_strong_periodicity', False),
      'dominant_period_hours': results['periodicity'].get('dominant_period_hours'),
      'predictability_score': results['predictability'].get('predictability_score'),
      'next_occurrence_prediction_hours': results['predictability'].get('next_expected_hours'),
      'hourly_concentration': results['temporal'].get('hourly_concentration'),
      'daily_concentration': results['temporal'].get('daily_concentration'),
    }

  # ----------------------------
  # NOVA ANÁLISE: Outliers Detalhada
  # ----------------------------
  def _analyze_outliers_detailed(self, intervals, render=True):
    """
    Análise completa de outliers com visualização
    """
    outliers_info = self._detect_outliers_comprehensive(intervals)
    
    if render:
      st.subheader("🚨 0. Análise de Outliers")
      
      col1, col2, col3 = st.columns(3)
      col1.metric("Total de Outliers", outliers_info['total_outliers'])
      col2.metric("Percentual", f"{outliers_info['outliers_percent']:.1f}%")
      col3.metric("Intervalos Normais", len(intervals) - outliers_info['total_outliers'])
      
      # Classificação da quantidade de outliers
      if outliers_info['outliers_percent'] > 20:
        st.error(f"🔴 **MUITOS OUTLIERS** ({outliers_info['outliers_percent']:.1f}%) - CV Robusto essencial!")
      elif outliers_info['outliers_percent'] > 15:
        st.warning(f"⚠️ **BASTANTES OUTLIERS** ({outliers_info['outliers_percent']:.1f}%) - CV Robusto recomendado")
      elif outliers_info['outliers_percent'] > 8:
        st.info(f"📊 **ALGUNS OUTLIERS** ({outliers_info['outliers_percent']:.1f}%) - CV Winsorizado pode ajudar")
      else:
        st.success(f"✅ **POUCOS OUTLIERS** ({outliers_info['outliers_percent']:.1f}%) - Dados bem comportados")
      
      # Detalhes por método
      with st.expander("📊 Detalhes dos Métodos de Detecção"):
        for method, info in outliers_info['methods'].items():
          st.write(f"**{method.upper()}**: {info['count']} outliers detectados")
      
      # Gráfico de intervalos com outliers destacados
      if outliers_info['total_outliers'] > 0:
        fig = go.Figure()
        
        # Intervalos normais
        normal_indices = [i for i in range(len(intervals)) if i not in outliers_info['outliers_indices']]
        fig.add_trace(go.Scatter(
          x=normal_indices,
          y=intervals[normal_indices],
          mode='markers',
          name='Normal',
          marker=dict(color='blue', size=8)
        ))
        
        # Outliers
        fig.add_trace(go.Scatter(
          x=outliers_info['outliers_indices'],
          y=outliers_info['outliers_values'],
          mode='markers',
          name='Outliers',
          marker=dict(color='red', size=12, symbol='x')
        ))
        
        # Linhas de referência
        fig.add_hline(y=np.median(intervals), line_dash="dash", 
                     line_color="green", annotation_text="Mediana")
        
        fig.update_layout(
          title="Intervalos: Normal vs Outliers",
          xaxis_title="Índice do Intervalo",
          yaxis_title="Intervalo (horas)",
          height=400
        )
        st.plotly_chart(fig, use_container_width=True, key=f'outliers_{self.alert_id}')
        
        # Tabela com os outliers
        if len(outliers_info['outliers_values']) <= 20:
          st.write("**🔴 Lista de Outliers:**")
          outliers_df = pd.DataFrame({
            'Índice': outliers_info['outliers_indices'],
            'Valor (horas)': [f"{v:.2f}" for v in outliers_info['outliers_values']]
          })
          st.dataframe(outliers_df, use_container_width=True)
    
    return outliers_info

  # ----------------------------
  # Métodos unificados (render opcional)
  # ----------------------------
  def _analyze_basic_statistics(self, intervals, render=True):
    # Calcular CVs com método adaptativo
    cv_adaptativo_result = self._cv_adaptativo(intervals)
    cv_usado = cv_adaptativo_result['cv']
    
    stats_dict = {
      'mean': float(np.mean(intervals)),
      'median': float(np.median(intervals)),
      'std': float(np.std(intervals)),
      'min': float(np.min(intervals)),
      'max': float(np.max(intervals)),
      'cv': cv_usado,
      'cv_classico': cv_adaptativo_result['cv_classico'],
      'cv_robusto': cv_adaptativo_result['cv_robusto'],
      'cv_winsorizado': cv_adaptativo_result['cv_winsorizado'],
      'cv_iqr': cv_adaptativo_result['cv_iqr'],
      'cv_metodo': cv_adaptativo_result['metodo'],
      'outliers_percent': cv_adaptativo_result['outliers_percent'],
      'diferenca_percentual': cv_adaptativo_result['diferenca_percentual'],
      'q25': float(np.percentile(intervals, 25)),
      'q75': float(np.percentile(intervals, 75)),
      'iqr': float(np.percentile(intervals, 75) - np.percentile(intervals, 25))
    }
    
    if render:
      st.subheader("📊 1. Estatísticas de Intervalos")
      col1, col2, col3, col4, col5 = st.columns(5)
      col1.metric("⏱️ Média", f"{stats_dict['mean']:.1f}h")
      col2.metric("📊 Mediana", f"{stats_dict['median']:.1f}h")
      col3.metric("📈 Desvio", f"{stats_dict['std']:.1f}h")
      col4.metric("⚡ Mínimo", f"{stats_dict['min']:.1f}h")
      col5.metric("🐌 Máximo", f"{stats_dict['max']:.1f}h")
      
      st.markdown("---")
      st.subheader("🎯 Coeficiente de Variação (CV) - Seleção Adaptativa")
      
      col1, col2, col3, col4 = st.columns(4)
      col1.metric("CV Clássico", f"{stats_dict['cv_classico']:.3f}")
      col2.metric("CV Robusto (MAD)", f"{stats_dict['cv_robusto']:.3f}")
      col3.metric("CV Winsorizado", f"{stats_dict['cv_winsorizado']:.3f}")
      col4.metric("CV IQR", f"{stats_dict['cv_iqr']:.3f}")
      
      # Mostrar qual foi escolhido
      st.success(f"**✅ CV Selecionado:** {stats_dict['cv_metodo']} = **{cv_usado:.3f}**")
      
      # Explicação da escolha
      outliers_pct = stats_dict['outliers_percent']
      diferenca_pct = stats_dict['diferenca_percentual']
      
      st.info(f"""
      **🔍 Critérios de Seleção:**
      - Outliers detectados: **{outliers_pct:.1f}%** (Z-score > 3)
      - Diferença Clássico vs Robusto: **{diferenca_pct:.1f}%**
      - Threshold Outliers: 20% → Robusto | 15% → Robusto | 8% → Winsorizado | <8% → Clássico
      - Threshold Diferença: >30% → Robusto
      """)
      
      # Gráfico comparativo
      fig = go.Figure(data=[
        go.Bar(name='Métodos de CV', 
               x=['Clássico', 'Robusto', 'Winsorizado', 'IQR'],
               y=[stats_dict['cv_classico'], stats_dict['cv_robusto'], 
                  stats_dict['cv_winsorizado'], stats_dict['cv_iqr']],
               marker_color=[
                 'green' if stats_dict['cv_metodo'] == 'Clássico' else 'lightgray',
                 'green' if stats_dict['cv_metodo'] == 'Robusto (MAD)' else 'lightgray',
                 'green' if stats_dict['cv_metodo'] == 'Winsorizado' else 'lightgray',
                 'lightgray'
               ],
               text=[f"{stats_dict['cv_classico']:.3f}", 
                     f"{stats_dict['cv_robusto']:.3f}",
                     f"{stats_dict['cv_winsorizado']:.3f}",
                     f"{stats_dict['cv_iqr']:.3f}"],
               textposition='outside')
      ])
      fig.update_layout(
        title="Comparação dos Métodos de CV (Verde = Selecionado)", 
        yaxis_title="Valor do CV", 
        height=300,
        showlegend=False
      )
      st.plotly_chart(fig, use_container_width=True, key=f'cv_comparison_{self.alert_id}')
      
    return stats_dict

  def _analyze_regularity(self, intervals, render=True):
    cv_result = self._cv_adaptativo(intervals)
    cv = cv_result['cv']
    
    if cv < 0.20:
      regularity_score, pattern_type, pattern_color = 95, "🟢 ALTAMENTE REGULAR", "green"
    elif cv < 0.40:
      regularity_score, pattern_type, pattern_color = 80, "🟢 REGULAR", "lightgreen"
    elif cv < 0.70:
      regularity_score, pattern_type, pattern_color = 60, "🟡 SEMI-REGULAR", "yellow"
    elif cv < 1.20:
      regularity_score, pattern_type, pattern_color = 35, "🟠 IRREGULAR", "orange"
    else:
      regularity_score, pattern_type, pattern_color = 15, "🔴 MUITO IRREGULAR", "red"

    if render:
      st.subheader("🎯 2. Regularidade (com CV Adaptativo)")
      col1, col2 = st.columns([3, 1])
      with col1:
        st.markdown(f"**Classificação:** {pattern_type}")
        st.write(f"**CV ({cv_result['metodo']}):** {cv:.3f} ({cv:.1%})")
        
        if cv_result['diferenca_percentual'] > 10:
          st.info(f"ℹ️ CV Clássico seria {cv_result['cv_classico']:.3f} (diferença de {cv_result['diferenca_percentual']:.1f}%)")
        
        if len(intervals) >= 3:
          _, p_value = stats.shapiro(intervals)
          if p_value > 0.05:
            st.success("📊 Distribuição aproximadamente normal")
          else:
            st.info("📊 Distribuição não-normal")
            
      with col2:
        fig = go.Figure(go.Indicator(
          mode="gauge+number",
          value=regularity_score,
          title={'text': "Regularidade"},
          gauge={'axis': {'range': [0, 100]}, 'bar': {'color': pattern_color}}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True, key=f'reg_gauge_{self.alert_id}')
        
    return {
      'cv': cv, 
      'regularity_score': regularity_score, 
      'type': pattern_type, 
      'cv_metodo': cv_result['metodo']
    }

  # [O resto dos métodos permanece igual - _analyze_periodicity, _analyze_autocorrelation, etc.]
  # Por brevidade, vou pular para os métodos modificados

  def _calculate_predictability(self, intervals, render=True):
    cv_result = self._cv_adaptativo(intervals)
    cv = cv_result['cv']
    
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
    mean_interval = float(np.mean(intervals))

    if render:
      st.subheader("🔮 11. Previsibilidade")
      col1, col2 = st.columns(2)
      col1.metric("Score", f"{predictability}%")
      col2.metric("Próxima Ocorrência", f"{mean_interval:.1f}h")
      if predictability > 70:
        st.success("✅ Altamente previsível")
      elif predictability > 50:
        st.info("📊 Moderadamente previsível")
      else:
        st.warning("⚠️ Pouco previsível")

    return {'predictability_score': int(predictability), 'next_expected_hours': mean_interval}

  # [Outros métodos intermediários permanecem iguais...]
  
  def _calculate_final_score_validated(self, results, df, intervals):
    regularity_score = results['regularity']['regularity_score'] * 0.25

    if results['periodicity'].get('has_strong_periodicity', False):
      periodicity_score = 100 * 0.25
    elif results['periodicity'].get('has_moderate_periodicity', False):
      periodicity_score = 50 * 0.25
    else:
      periodicity_score = 0 * 0.25

    predictability_score = results['predictability']['predictability_score'] * 0.15

    hourly_conc = results['temporal']['hourly_concentration']
    daily_conc = results['temporal']['daily_concentration']
    concentration_score = 0
    if hourly_conc > 60 or daily_conc > 60:
      concentration_score = 100 * 0.20
    elif hourly_conc > 40 or daily_conc > 40:
      concentration_score = 60 * 0.20
    elif hourly_conc > 30 or daily_conc > 30:
      concentration_score = 30 * 0.20

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

    final_score = (regularity_score + periodicity_score + predictability_score + concentration_score + frequency_score)

    if final_score >= 70 and total_occurrences >= 10:
      classification = "🔴 REINCIDENTE CRÍTICO (P1)"
    elif final_score >= 50 and total_occurrences >= 5:
      classification = "🟠 PARCIALMENTE REINCIDENTE (P2)"
    elif final_score >= 35:
      classification = "🟡 PADRÃO DETECTÁVEL (P3)"
    else:
      classification = "🟢 NÃO REINCIDENTE (P4)"

    return round(float(final_score), 2), classification

  def _final_classification(self, results, df, intervals):
    st.markdown("---")
    st.header("🎯 CLASSIFICAÇÃO FINAL")
    final_score, classification = self._calculate_final_score_validated(results, df, intervals)

    if final_score >= 70:
      level, color, priority, = "CRÍTICO", "red", "P1"
    elif final_score >= 50:
      level, color, priority, = "ALTO", "orange", "P2"
    elif final_score >= 35:
      level, color, priority, = "MÉDIO", "yellow", "P3" 
    else:
      level, color, priority, = "BAIXO", "green", "P4"

    col1, col2 = st.columns([2, 1])
    with col1:
      st.markdown(f"### {classification}")
      st.markdown(f"**Nível:** {level} | **Prioridade:** {priority}")
      st.metric("Score de Reincidência", f"{final_score:.0f}/100")
      
      # Mostrar qual CV foi usado
      cv_metodo = results['basic_stats'].get('cv_metodo', 'clássico')
      cv_usado = results['basic_stats'].get('cv', 0)
      st.success(f"**✅ Método de CV utilizado no score:** {cv_metodo} = {cv_usado:.3f}")
      
      st.markdown("#### 📊 Breakdown dos Critérios VALIDADOS (com CV Robusto)")
      total_occurrences = len(df)
      period_days = (df['created_on'].max() - df['created_on'].min()).days + 1
      freq_per_week = (total_occurrences / period_days * 7) if period_days > 0 else 0
      regularity_pts = results['regularity']['regularity_score'] * 0.25
      periodicity_pts = (100 * 0.25) if results['periodicity'].get('has_strong_periodicity', False) else 0
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

      breakdown = {
        f'1. Regularidade (25%) - CV {cv_metodo}': regularity_pts,
        '2. Periodicidade (25%)': periodicity_pts,
        '3. Previsibilidade (15%)': predictability_pts,
        '4. Concentração Temporal (20%)': concentration_pts,
        '5. Frequência Absoluta (15%)': frequency_pts,
      }

      for criterion, points in breakdown.items():
        st.write(f"• {criterion}: **{points:.1f} pts**")


    with col2:
      fig = go.Figure(go.Indicator(mode="gauge+number", value=final_score, title={'text': "Score Final"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}}))
      fig.update_layout(height=300)
      st.plotly_chart(fig, use_container_width=True, key=f'final_gauge_{self.alert_id}')

    # Exportar resumo
    st.markdown("---")
    export_data = {
      'u_alert_id': self.alert_id,
      'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      'score': final_score,
      'classificacao': classification,
      'nivel': level,
      'prioridade': priority,
      'total_occurrences': len(df),
      'freq_per_week': freq_per_week,
      'cv': results['basic_stats']['cv'],
      'cv_classico': results['basic_stats']['cv_classico'],
      'cv_robusto': results['basic_stats']['cv_robusto'],
      'cv_metodo': results['basic_stats']['cv_metodo'],
      'outliers_percent': results['basic_stats']['outliers_percent'],
      'diferenca_cv_percent': results['basic_stats']['diferenca_percentual'],
      'regularidade': results['regularity']['regularity_score'],
      'periodicidade': results['periodicity'].get('has_strong_periodicity', False),
      'previsibilidade': results['predictability']['predictability_score'],
      'concentracao_horaria': results['temporal']['hourly_concentration'],
      'concentracao_diaria': results['temporal']['daily_concentration'],
      'bursts_detected': results['bursts']['has_bursts'],
      'n_bursts': results['bursts']['n_bursts'],
    }
    export_df = pd.DataFrame([export_data])
    csv = export_df.to_csv(index=False)
    st.download_button("⬇️ Exportar Relatório Completo", csv, f"reincidencia_{self.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)


# ============================================================
# StreamlitAlertAnalyzer: UI glue + batch processing
# ============================================================
class StreamlitAlertAnalyzer:
  def __init__(self):
    self.df_original = None
    self.df = None
    self.dates = None
    self.alert_id = None

  def load_data(self, uploaded_file):
    try:
      df_raw = pd.read_csv(uploaded_file)
      st.success(f"✅ Arquivo carregado: {len(df_raw)} registros")
      with st.expander("📋 Preview"):
        st.write(f"**Colunas:** {list(df_raw.columns)}")
        st.dataframe(df_raw.head())
      if 'created_on' not in df_raw.columns or 'u_alert_id' not in df_raw.columns:
        st.error("❌ Colunas obrigatórias: 'created_on' e 'u_alert_id'")
        return False
      df_raw['created_on'] = pd.to_datetime(df_raw['created_on'])
      df_raw = df_raw.dropna(subset=['created_on']).sort_values(['u_alert_id', 'created_on']).reset_index(drop=True)
      self.df_original = df_raw
      st.sidebar.write(f"**IDs:** {len(df_raw['u_alert_id'].unique())}")
      return True
    except Exception as e:
      st.error(f"❌ Erro: {e}")
      return False

  def prepare_individual_analysis(self, alert_id):
    df_filtered = self.df_original[self.df_original['u_alert_id'] == alert_id].copy()
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

  def complete_analysis_all_u_alert_id(self, progress_bar=None):
    try:
      if self.df_original is None or len(self.df_original) == 0:
        st.error("❌ Dados não carregados")
        return None
      u_alert_id_list = list(self.df_original['u_alert_id'].unique())
      total = len(u_alert_id_list)
      use_mp = total > 20
      if use_mp:
        n_processes = min(cpu_count(), total, 8)
        st.info(f"🚀 Usando {n_processes} processos para {total} alertas")
        chunk_size = max(1, total // n_processes)
        chunks = [u_alert_id_list[i:i + chunk_size] for i in range(0, total, chunk_size)]
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
        for idx, u_alert_id in enumerate(u_alert_id_list):
          if progress_bar:
            progress_bar.progress((idx + 1) / total, text=f"{idx + 1}/{total}")
          result = analyze_single_u_alert_id_recurrence(u_alert_id, self.df_original)
          if result:
            all_results.append(result)
        return pd.DataFrame(all_results)
    except Exception as e:
      st.error(f"Erro: {e}")
      import traceback
      st.error(traceback.format_exc())
      return None

  def show_basic_stats(self):
    st.header("📊 Estatísticas Básicas")
    total = len(self.df)
    period_days = (self.dates.max() - self.dates.min()).days + 1
    avg_per_day = total / period_days if period_days > 0 else 0
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


# ============================================================
# MAIN
# ============================================================
def main():
  st.title("🚨 Analisador de Alertas com CV Robusto 🛡️")
  st.markdown("### 2 modos: Individual e Completa + CSV (com critérios validados e CV adaptativo)")
  st.sidebar.header("⚙️ Configurações")
  analysis_mode = st.sidebar.selectbox("🎯 Modo de Análise", ["🔍 Individual", "📊 Completa + CSV"])
  uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=['csv'])

  if uploaded_file:
    analyzer = StreamlitAlertAnalyzer()
    if analyzer.load_data(uploaded_file):
      if analysis_mode == "🔍 Individual":
        id_counts = analyzer.df_original['u_alert_id'].value_counts()
        id_options = [f"{uid} ({count})" for uid, count in id_counts.items()]
        selected = st.sidebar.selectbox("Short CI", id_options)
        selected_id = selected.split(" (")[0]
        if st.sidebar.button("🚀 Analisar", type="primary"):
          if analyzer.prepare_individual_analysis(selected_id):
            st.success(f"Analisando: {selected_id}")
            tab1, tab2 = st.tabs(["📊 Básico", "🔄 Reincidência"])
            with tab1:
              analyzer.show_basic_stats()
            with tab2:
              recurrence_analyzer = AdvancedRecurrenceAnalyzer(analyzer.df, selected_id)
              recurrence_analyzer.analyze()

      elif analysis_mode == "📊 Completa + CSV":
        st.subheader("📊 Análise Completa COM CV ROBUSTO")
        if st.sidebar.button("🚀 Executar", type="primary"):
          st.info("⏱️ Processando com multiprocessing e CV adaptativo...")
          progress_bar = st.progress(0)
          df_consolidated = analyzer.complete_analysis_all_u_alert_id(progress_bar)
          progress_bar.empty()
          if df_consolidated is not None and len(df_consolidated) > 0:
            st.success(f"✅ {len(df_consolidated)} alertas processados com CV Robusto!")
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
            
            # Estatísticas de CV
            st.markdown("---")
            st.subheader("🎯 Estatísticas de CV Robusto")
            if 'cv_metodo' in df_consolidated.columns:
              metodo_counts = df_consolidated['cv_metodo'].value_counts()
              col1, col2, col3 = st.columns(3)
              for idx, (metodo, count) in enumerate(metodo_counts.items()):
                if idx == 0:
                  col1.metric(f"{metodo}", f"{count} ({count/len(df_consolidated)*100:.1f}%)")
                elif idx == 1:
                  col2.metric(f"{metodo}", f"{count} ({count/len(df_consolidated)*100:.1f}%)")
                elif idx == 2:
                  col3.metric(f"{metodo}", f"{count} ({count/len(df_consolidated)*100:.1f}%)")
            
            st.subheader("Dataframe Completo")
            st.dataframe(df_consolidated, use_container_width=True)
            st.markdown("---")
            st.subheader("📥 Exportar")
            col1, col2 = st.columns(2)
            csv_full = df_consolidated.to_csv(index=False)
            col1.download_button("⬇️ CSV Completo", csv_full, f"completo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
            summary_cols = ['u_alert_id', 'score', 'classification', 'total_occurrences', 'cv', 'cv_robusto', 'cv_metodo', 'outliers_percent', 'diferenca_cv_percent']
            available_summary = [col for col in summary_cols if col in df_consolidated.columns]
            summary = df_consolidated[available_summary].copy()
            csv_summary = summary.to_csv(index=False)
            col2.download_button("⬇️ CSV Resumido", csv_summary, f"resumo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
  else:
    st.info("👆 Faça upload de um CSV")

if __name__ == "__main__":
  main()
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
    page_title="Analisador de Alertas - Otimizado",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FUNÇÕES AUXILIARES PARA MULTIPROCESSING
# ============================================================

def analyze_single_short_ci_recurrence(short_ci, df_original):
    """
    Função auxiliar para análise de reincidência de um único short_ci.
    Usada em multiprocessing.
    """
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
        
        # Criar analisador e executar análise silenciosa
        analyzer = AdvancedRecurrenceAnalyzer(df_ci, short_ci)
        return analyzer.analyze_silent()
    
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
# CLASSE PRINCIPAL DE ANÁLISE DE REINCIDÊNCIA (OTIMIZADA)
# ============================================================

class AdvancedRecurrenceAnalyzer:
    """Analisador otimizado de padrões de reincidência"""
    
    def __init__(self, df, alert_id):
        self.df = df.copy() if df is not None else None
        self.alert_id = alert_id
        self.cache = {}
    
    def _prepare_data(self):
        """Preparação vetorizada dos dados"""
        if self.df is None or len(self.df) < 3:
            return None
        
        df = self.df.sort_values('created_on').copy()
        
        # Vetorizar todas as operações de timestamp
        df['timestamp'] = df['created_on'].astype('int64') // 10**9
        df['time_diff_seconds'] = df['timestamp'].diff()
        df['time_diff_hours'] = df['time_diff_seconds'] / 3600
        
        # Extrair componentes temporais de uma vez
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
    
    def analyze(self):
        """Método principal com interface Streamlit"""
        st.header("🔄 Análise de Reincidência Temporal")
        
        df = self._prepare_data()
        if df is None:
            st.warning("⚠️ Dados insuficientes (mínimo 3 ocorrências).")
            return
        
        st.info(f"📊 Analisando **{len(df)}** ocorrências do Short CI: **{self.alert_id}**")
        
        intervals_hours = df['time_diff_hours'].dropna().values
        if len(intervals_hours) < 2:
            st.warning("⚠️ Intervalos insuficientes.")
            return
        
        # Executar análises
        results = self._run_all_analyses(df, intervals_hours, silent=False)
        
        # Classificação final
        self._final_classification(results, df, intervals_hours)
    
    def analyze_silent(self):
        """Análise silenciosa para processamento em lote"""
        df = self._prepare_data()
        if df is None or len(df) < 3:
            return None
        
        intervals_hours = df['time_diff_hours'].dropna().values
        if len(intervals_hours) < 2:
            return None
        
        # Executar análises sem interface
        results = self._run_all_analyses(df, intervals_hours, silent=True)
        
        # Calcular score final
        final_score, classification = self._calculate_final_score(results)
        
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
            'next_occurrence_prediction_hours': results['predictability']['next_expected_hours']
        }
    
    def _run_all_analyses(self, df, intervals, silent=False):
        """Executa todas as análises essenciais - versão unificada"""
        results = {}
        
        # 1. Estatísticas Básicas
        results['basic_stats'] = self._analyze_basic_statistics(intervals, silent)
        
        # 2. Regularidade
        results['regularity'] = self._analyze_regularity(intervals, silent)
        
        # 3. Periodicidade
        results['periodicity'] = self._analyze_periodicity(intervals, silent)
        
        # 4. Autocorrelação
        results['autocorr'] = self._analyze_autocorrelation(intervals, silent)
        
        # 5. Previsibilidade
        results['predictability'] = self._calculate_predictability(intervals, silent)
        
        # 6. Markov
        results['markov'] = self._analyze_markov_chains(intervals, silent)
        
        # 7. Aleatoriedade
        results['randomness'] = self._advanced_randomness_tests(intervals, silent)
        
        # 8. Estabilidade (se tiver dados suficientes)
        if len(intervals) >= 10:
            results['stability'] = self._analyze_stability(intervals, df, silent)
        else:
            results['stability'] = {'is_stable': True, 'stability_score': 50}
        
        return results
    
    # ============================================================
    # ANÁLISES UNIFICADAS (com parâmetro silent)
    # ============================================================
    
    def _analyze_basic_statistics(self, intervals, silent=False):
        """Estatísticas básicas - versão unificada"""
        stats_dict = {
            'mean': np.mean(intervals),
            'median': np.median(intervals),
            'std': np.std(intervals),
            'min': np.min(intervals),
            'max': np.max(intervals),
            'cv': np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        }
        
        if not silent:
            st.subheader("📊 1. Estatísticas de Intervalos")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("⏱️ Média", f"{stats_dict['mean']:.1f}h")
            col2.metric("📊 Mediana", f"{stats_dict['median']:.1f}h")
            col3.metric("📈 Desvio", f"{stats_dict['std']:.1f}h")
            col4.metric("⚡ Mínimo", f"{stats_dict['min']:.1f}h")
            col5.metric("🐌 Máximo", f"{stats_dict['max']:.1f}h")
        
        return stats_dict
    
    def _analyze_regularity(self, intervals, silent=False):
        """Análise de regularidade - versão unificada"""
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        
        # Classificação simplificada e clara
        if cv < 0.20:  # Muito regular
            regularity_score = 95
            pattern_type = "🟢 ALTAMENTE REGULAR"
            pattern_color = "green"
        elif cv < 0.40:  # Regular
            regularity_score = 80
            pattern_type = "🟢 REGULAR"
            pattern_color = "lightgreen"
        elif cv < 0.70:  # Semi-regular
            regularity_score = 60
            pattern_type = "🟡 SEMI-REGULAR"
            pattern_color = "yellow"
        elif cv < 1.20:  # Irregular
            regularity_score = 35
            pattern_type = "🟠 IRREGULAR"
            pattern_color = "orange"
        else:  # Muito irregular
            regularity_score = 15
            pattern_type = "🔴 MUITO IRREGULAR"
            pattern_color = "red"
        
        if not silent:
            st.subheader("🎯 2. Regularidade")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Classificação:** {pattern_type}")
                st.write(f"**CV (Coeficiente de Variação):** {cv:.2%}")
                
                # Interpretação
                if cv < 0.20:
                    st.success("✅ Intervalos muito consistentes - padrão altamente previsível")
                elif cv < 0.40:
                    st.info("📊 Intervalos razoavelmente consistentes - padrão previsível")
                elif cv < 0.70:
                    st.warning("⚠️ Intervalos moderadamente variáveis")
                else:
                    st.error("❌ Intervalos muito variáveis - padrão imprevisível")
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=regularity_score,
                    title={'text': "Score"},
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
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True, key='reg_gauge')
        
        return {'cv': cv, 'regularity_score': regularity_score, 'type': pattern_type}
    
    def _analyze_periodicity(self, intervals, silent=False):
        """Análise de periodicidade via FFT - versão unificada"""
        if len(intervals) < 4:
            return {
                'has_strong_periodicity': False,
                'has_moderate_periodicity': False,
                'dominant_period_hours': None
            }
        
        try:
            # FFT otimizada
            N = len(intervals)
            yf = fft(intervals)
            xf = fftfreq(N, d=1)[:N//2]
            power = 2.0/N * np.abs(yf[:N//2])
            
            if len(power) > 1:
                # Encontrar pico dominante (ignorando frequência 0)
                peak_idx = np.argmax(power[1:]) + 1
                dominant_freq = xf[peak_idx]
                dominant_period = 1/dominant_freq if dominant_freq != 0 else None
                
                # Converter para horas
                mean_interval = np.mean(intervals)
                dominant_period_hours = dominant_period * mean_interval if dominant_period else None
                
                # Força da periodicidade
                peak_power = power[peak_idx]
                mean_power = np.mean(power)
                strength_ratio = peak_power / mean_power if mean_power > 0 else 0
                
                # Classificação revista
                has_strong = strength_ratio > 3.0  # Muito forte
                has_moderate = 1.5 < strength_ratio <= 3.0  # Moderado
                
                if not silent and (has_strong or has_moderate):
                    st.subheader("🔍 3. Periodicidade (FFT)")
                    if has_strong:
                        st.success(f"✅ **Periodicidade FORTE detectada**")
                    else:
                        st.info(f"📊 **Periodicidade MODERADA detectada**")
                    
                    if dominant_period_hours:
                        time_str = f"{dominant_period_hours:.1f}h" if dominant_period_hours < 24 else f"{dominant_period_hours/24:.1f} dias"
                        st.write(f"• Período dominante: **~{time_str}**")
                        st.write(f"• Força do padrão: **{strength_ratio:.1f}x** acima da média")
                
                return {
                    'has_strong_periodicity': has_strong,
                    'has_moderate_periodicity': has_moderate,
                    'dominant_period_hours': dominant_period_hours,
                    'strength_ratio': strength_ratio
                }
        except Exception:
            pass
        
        if not silent:
            st.subheader("🔍 3. Periodicidade")
            st.info("📊 Nenhuma periodicidade clara detectada")
        
        return {
            'has_strong_periodicity': False,
            'has_moderate_periodicity': False,
            'dominant_period_hours': None
        }
    
    def _analyze_autocorrelation(self, intervals, silent=False):
        """Análise de autocorrelação - versão unificada"""
        if len(intervals) < 5:
            return {'max_autocorr': 0, 'has_autocorr': False}
        
        try:
            # Calcular autocorrelação para lags 1-20
            max_lag = min(len(intervals) // 2, 20)
            autocorr_values = []
            
            for lag in range(1, max_lag + 1):
                if lag < len(intervals):
                    corr = np.corrcoef(intervals[:-lag], intervals[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr_values.append(abs(corr))
            
            max_autocorr = max(autocorr_values) if autocorr_values else 0
            has_autocorr = max_autocorr > 0.3  # Threshold revisado
            
            if not silent and has_autocorr:
                st.subheader("📈 4. Autocorrelação")
                st.success(f"✅ **Autocorrelação significativa detectada**: {max_autocorr:.2f}")
                st.write("• Eventos correlacionados indicam padrão recorrente")
            
            return {'max_autocorr': max_autocorr, 'has_autocorr': has_autocorr}
        
        except Exception:
            return {'max_autocorr': 0, 'has_autocorr': False}
    
    def _calculate_predictability(self, intervals, silent=False):
        """Score de previsibilidade - versão unificada"""
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        
        # Score baseado em CV (revisado para ser mais rigoroso)
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
        
        if not silent:
            st.subheader("🔮 5. Previsibilidade")
            col1, col2 = st.columns(2)
            col1.metric("Score de Previsibilidade", f"{predictability}%")
            col2.metric("Próxima Ocorrência (estimada)", f"{mean_interval:.1f}h")
            
            if predictability > 70:
                st.success("✅ Padrão altamente previsível")
            elif predictability > 50:
                st.info("📊 Padrão moderadamente previsível")
            else:
                st.warning("⚠️ Padrão pouco previsível")
        
        return {
            'predictability_score': predictability,
            'next_expected_hours': mean_interval
        }
    
    def _analyze_markov_chains(self, intervals, silent=False):
        """Análise de Cadeias de Markov - versão unificada"""
        if len(intervals) < 5:
            return {'markov_score': 0}
        
        try:
            # Discretizar em 3 estados
            bins = np.percentile(intervals, [0, 33, 67, 100])
            states = np.digitize(intervals, bins[1:-1])
            
            n_states = 3
            transition_matrix = np.zeros((n_states, n_states))
            
            # Construir matriz de transição
            for i in range(len(states) - 1):
                current = states[i]
                next_state = states[i + 1]
                transition_matrix[current, next_state] += 1
            
            # Normalizar
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            transition_matrix = transition_matrix / row_sums
            
            # Score baseado em previsibilidade das transições
            max_probs = transition_matrix.max(axis=1)
            markov_score = np.mean(max_probs) * 100
            
            if not silent and markov_score > 50:
                st.subheader("🔗 6. Padrão Markoviano")
                st.info(f"📊 Score Markoviano: **{markov_score:.1f}%**")
                if markov_score > 60:
                    st.success("✅ Forte padrão de transições - estado atual prevê o próximo")
            
            return {'markov_score': markov_score}
        
        except Exception:
            return {'markov_score': 0}
    
    def _advanced_randomness_tests(self, intervals, silent=False):
        """Testes de aleatoriedade - versão unificada e simplificada"""
        if len(intervals) < 5:
            return {'overall_randomness_score': 50}
        
        try:
            randomness_scores = []
            
            # 1. Runs Test simplificado
            median = np.median(intervals)
            runs = np.diff(intervals > median).sum() + 1
            expected_runs = len(intervals) / 2
            runs_deviation = abs(runs - expected_runs) / expected_runs
            runs_score = min(runs_deviation * 100, 100)
            randomness_scores.append(runs_score)
            
            # 2. Variação no CV ao longo do tempo
            if len(intervals) > 10:
                mid = len(intervals) // 2
                cv1 = np.std(intervals[:mid]) / np.mean(intervals[:mid])
                cv2 = np.std(intervals[mid:]) / np.mean(intervals[mid:])
                cv_stability = abs(cv1 - cv2) / max(cv1, cv2) * 100
                randomness_scores.append(cv_stability)
            
            # Score geral de aleatoriedade
            overall_randomness = np.mean(randomness_scores)
            
            if not silent:
                st.subheader("🎲 7. Testes de Aleatoriedade")
                determinism = 100 - overall_randomness
                st.metric("Determinismo", f"{determinism:.1f}%")
                
                if determinism > 70:
                    st.success("✅ Comportamento determinístico - padrão estruturado")
                elif determinism > 40:
                    st.info("📊 Comportamento misto")
                else:
                    st.warning("⚠️ Comportamento aleatório")
            
            return {'overall_randomness_score': overall_randomness}
        
        except Exception:
            return {'overall_randomness_score': 50}
    
    def _analyze_stability(self, intervals, df, silent=False):
        """Análise de estabilidade - versão unificada"""
        if len(intervals) < 10:
            return {'is_stable': True, 'stability_score': 50}
        
        try:
            # Dividir em primeira e segunda metade
            mid = len(intervals) // 2
            first_half = intervals[:mid]
            second_half = intervals[mid:]
            
            # Teste t
            _, p_value = stats.ttest_ind(first_half, second_half)
            is_stable = p_value > 0.05
            
            # Score baseado em diferença de médias
            mean_diff = abs(np.mean(second_half) - np.mean(first_half))
            drift_pct = (mean_diff / np.mean(first_half)) * 100 if np.mean(first_half) > 0 else 0
            
            # Score de estabilidade (quanto menor o drift, maior o score)
            stability_score = max(0, 100 - drift_pct)
            
            if not silent:
                st.subheader("📊 8. Estabilidade")
                col1, col2 = st.columns(2)
                col1.metric("Score de Estabilidade", f"{stability_score:.1f}%")
                col2.metric("Drift", f"{drift_pct:.1f}%")
                
                if is_stable and drift_pct < 20:
                    st.success("✅ Padrão estável no tempo")
                elif drift_pct < 50:
                    st.info("📊 Padrão moderadamente estável")
                else:
                    st.warning("⚠️ Padrão instável - variação significativa")
            
            return {
                'is_stable': is_stable,
                'stability_score': stability_score,
                'drift_pct': drift_pct
            }
        
        except Exception:
            return {'is_stable': True, 'stability_score': 50}
    
    # ============================================================
    # CLASSIFICAÇÃO FINAL REVISADA
    # ============================================================
    
    def _calculate_final_score(self, results):
        """
        Calcula score final baseado em critérios essenciais REVISADOS
        Foco em determinar REINCIDÊNCIA de forma objetiva
        """
        scores = {
            # 1. REGULARIDADE (25%) - O MAIS IMPORTANTE
            'regularity': results['regularity']['regularity_score'] * 0.25,
            
            # 2. PERIODICIDADE (25%) - MUITO IMPORTANTE
            'periodicity': (
                100 if results['periodicity']['has_strong_periodicity'] else
                50 if results['periodicity'].get('has_moderate_periodicity', False) else
                0
            ) * 0.25,
            
            # 3. PREVISIBILIDADE (20%)
            'predictability': results['predictability']['predictability_score'] * 0.20,
            
            # 4. DETERMINISMO (15%)
            'determinism': (100 - results['randomness']['overall_randomness_score']) * 0.15,
            
            # 5. AUTOCORRELAÇÃO (10%)
            'autocorrelation': (results['autocorr']['max_autocorr'] * 100) * 0.10,
            
            # 6. ESTABILIDADE (5%) - Reduzido pois é menos importante
            'stability': results.get('stability', {}).get('stability_score', 50) * 0.05
        }
        
        final_score = sum(scores.values())
        
        # Classificação REVISADA com thresholds mais rigorosos
        if final_score >= 75:
            classification = "🔴 REINCIDENTE CRÍTICO (P1)"
        elif final_score >= 55:
            classification = "🟠 PARCIALMENTE REINCIDENTE (P2)"
        elif final_score >= 35:
            classification = "🟡 PADRÃO DETECTÁVEL (P3)"
        else:
            classification = "🟢 NÃO REINCIDENTE (P4)"
        
        return round(final_score, 2), classification
    
    def _final_classification(self, results, df, intervals):
        """Classificação final com interface"""
        st.markdown("---")
        st.header("🎯 CLASSIFICAÇÃO FINAL")
        
        final_score, classification = self._calculate_final_score(results)
        
        # Determinar nível e cor
        if final_score >= 75:
            level = "CRÍTICO"
            color = "red"
            priority = "P1"
            recommendation = "**Ação Imediata:** Criar automação, runbook e investigar causa raiz"
        elif final_score >= 55:
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
            recommendation = "**Ação:** Análise caso a caso - possível comportamento normal"
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### {classification}")
            st.markdown(f"**Nível:** {level} | **Prioridade:** {priority}")
            st.metric("Score de Reincidência", f"{final_score:.0f}/100")
            
            st.markdown("#### 📊 Breakdown do Score")
            breakdown_data = {
                '✅ Regularidade (25%)': results['regularity']['regularity_score'] * 0.25,
                '🔍 Periodicidade (25%)': (100 if results['periodicity']['has_strong_periodicity'] else 
                                          50 if results['periodicity'].get('has_moderate_periodicity', False) else 0) * 0.25,
                '🔮 Previsibilidade (20%)': results['predictability']['predictability_score'] * 0.20,
                '🎲 Determinismo (15%)': (100 - results['randomness']['overall_randomness_score']) * 0.15,
                '📈 Autocorrelação (10%)': (results['autocorr']['max_autocorr'] * 100) * 0.10,
                '📊 Estabilidade (5%)': results.get('stability', {}).get('stability_score', 50) * 0.05
            }
            
            for criterion, points in breakdown_data.items():
                st.write(f"• {criterion}: **{points:.1f} pts**")
            
            st.info(recommendation)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_score,
                title={'text': "Score Final", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 35], 'color': "lightgray"},
                        {'range': [35, 55], 'color': "lightyellow"},
                        {'range': [55, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key='final_gauge')
        
        # Predição se score alto
        if final_score >= 55:
            st.markdown("---")
            st.subheader("🔮 Predição de Próxima Ocorrência")
            
            last_alert = df['created_on'].max()
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            pred_time = last_alert + pd.Timedelta(hours=mean_interval)
            conf_interval = 1.96 * std_interval
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Próxima Ocorrência (95%)", pred_time.strftime('%d/%m %H:%M'))
            col2.metric("Intervalo Esperado", f"{mean_interval:.1f}h")
            col3.metric("Margem de Erro", f"± {conf_interval:.1f}h")
        
        # Exportar
        st.markdown("---")
        export_data = {
            'short_ci': self.alert_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'score': final_score,
            'classificacao': classification,
            'nivel': level,
            'prioridade': priority,
            'cv': results['basic_stats']['cv'],
            'regularidade': results['regularity']['regularity_score'],
            'periodicidade': results['periodicity']['has_strong_periodicity'],
            'previsibilidade': results['predictability']['predictability_score'],
            'recomendacao': recommendation
        }
        
        export_df = pd.DataFrame([export_data])
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="⬇️ Exportar Relatório",
            data=csv,
            file_name=f"reincidencia_{self.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# FUNÇÕES AUXILIARES DE AGRUPAMENTO (mantidas)
# ============================================================

def identify_alert_groups(alert_data, max_gap_hours=24, min_group_size=3, spike_threshold_multiplier=5):
    """Identifica grupos de alertas contínuos"""
    if len(alert_data) == 0:
        return alert_data, []
    
    alert_data = alert_data.sort_values('created_on').reset_index(drop=True)
    alert_data['time_diff_hours'] = alert_data['created_on'].diff().dt.total_seconds() / 3600
    alert_data['date'] = alert_data['created_on'].dt.date
    unique_dates = alert_data['date'].nunique()
    
    if unique_dates == 1:
        alert_data['group_id'] = -1
        alert_data['is_isolated'] = True
        return alert_data, []
    
    alert_data['group_id'] = -1
    current_group = 0
    group_start_idx = 0
    
    for i in range(len(alert_data)):
        if i == 0:
            continue
        
        gap = alert_data.loc[i, 'time_diff_hours']
        
        if gap > max_gap_hours:
            group_size = i - group_start_idx
            if group_size >= min_group_size:
                alert_data.loc[group_start_idx:i-1, 'group_id'] = current_group
                current_group += 1
            group_start_idx = i
    
    group_size = len(alert_data) - group_start_idx
    if group_size >= min_group_size:
        alert_data.loc[group_start_idx:, 'group_id'] = current_group
    
    daily_counts = alert_data.groupby('date').size()
    avg_daily = daily_counts.mean()
    spike_threshold = avg_daily * spike_threshold_multiplier
    spike_dates = daily_counts[daily_counts > spike_threshold].index
    
    if len(spike_dates) > 0:
        alert_data.loc[alert_data['date'].isin(spike_dates), 'group_id'] = -1
    
    alert_data['is_isolated'] = alert_data['group_id'] == -1
    
    groups_info = []
    for group_id in alert_data[alert_data['group_id'] >= 0]['group_id'].unique():
        group_data = alert_data[alert_data['group_id'] == group_id]
        groups_info.append({
            'group_id': int(group_id),
            'size': len(group_data),
            'start_time': group_data['created_on'].min(),
            'end_time': group_data['created_on'].max(),
            'duration_hours': (group_data['created_on'].max() - group_data['created_on'].min()).total_seconds() / 3600
        })
    
    return alert_data, groups_info


def classify_alert_pattern(alert_data, max_gap_hours=24, min_group_size=3, spike_threshold_multiplier=5):
    """Classifica padrão do alerta"""
    n = len(alert_data)
    if n == 0:
        return {
            'pattern': 'isolated',
            'reason': 'Sem ocorrências',
            'occurrences': 0,
            'num_groups': 0,
            'isolated_occurrences': 0,
            'grouped_occurrences': 0,
            'groups_info': [],
            'unique_days': 0
        }
    
    unique_days = alert_data['created_on'].dt.date.nunique()
    
    if unique_days == 1:
        return {
            'pattern': 'isolated',
            'reason': f'Todos os {n} alertas em um único dia',
            'occurrences': n,
            'num_groups': 0,
            'isolated_occurrences': n,
            'grouped_occurrences': 0,
            'groups_info': [],
            'unique_days': 1
        }
    
    alert_data_processed, groups_info = identify_alert_groups(
        alert_data, max_gap_hours, min_group_size, spike_threshold_multiplier
    )
    
    num_groups = len(groups_info)
    isolated_count = alert_data_processed['is_isolated'].sum()
    grouped_count = n - isolated_count
    isolated_pct = (isolated_count / n) * 100
    
    if num_groups == 0:
        pattern = 'isolated'
        reason = f'Nenhum grupo ({n} isolados)'
    elif isolated_pct > 70:
        pattern = 'isolated'
        reason = f'{isolated_pct:.0f}% isolados'
    elif num_groups >= 2:
        pattern = 'continuous'
        reason = f'{num_groups} grupos ({grouped_count} agrupados)'
    else:
        pattern = 'isolated'
        reason = f'Padrão inconsistente'
    
    return {
        'pattern': pattern,
        'reason': reason,
        'occurrences': n,
        'num_groups': num_groups,
        'isolated_occurrences': int(isolated_count),
        'grouped_occurrences': int(grouped_count),
        'groups_info': groups_info,
        'unique_days': unique_days
    }


def process_single_alert(alert_id, df_original, max_gap_hours=24, min_group_size=3, spike_threshold_multiplier=5):
    """Processa um único alerta para análise global"""
    try:
        df_alert = df_original[df_original['short_ci'] == alert_id].copy()
        if len(df_alert) < 1:
            return None
        
        pattern_info = classify_alert_pattern(df_alert, max_gap_hours, min_group_size, spike_threshold_multiplier)
        
        df_alert['hour'] = df_alert['created_on'].dt.hour
        df_alert['day_of_week'] = df_alert['created_on'].dt.dayofweek
        df_alert['is_weekend'] = df_alert['day_of_week'].isin([5, 6])
        df_alert['is_business_hours'] = (df_alert['hour'] >= 9) & (df_alert['hour'] <= 17)
        df_alert = df_alert.sort_values('created_on')
        intervals_hours = df_alert['created_on'].diff().dt.total_seconds() / 3600
        intervals_hours = intervals_hours.dropna()
        
        period_days = (df_alert['created_on'].max() - df_alert['created_on'].min()).days + 1
        
        metrics = {
            'alert_id': alert_id,
            'pattern_type': pattern_info['pattern'],
            'pattern_reason': pattern_info['reason'],
            'total_ocorrencias': pattern_info['occurrences'],
            'num_grupos': pattern_info['num_groups'],
            'alertas_isolados': pattern_info['isolated_occurrences'],
            'alertas_agrupados': pattern_info['grouped_occurrences'],
            'pct_isolados': (pattern_info['isolated_occurrences'] / pattern_info['occurrences'] * 100) 
                           if pattern_info['occurrences'] > 0 else 0,
            'unique_days': pattern_info['unique_days'],
            'periodo_dias': period_days,
            'freq_dia': len(df_alert) / period_days if period_days > 0 else 0,
            'freq_semana': (len(df_alert) / period_days * 7) if period_days > 0 else 0,
            'freq_mes': (len(df_alert) / period_days * 30) if period_days > 0 else 0,
            'intervalo_medio_h': intervals_hours.mean() if len(intervals_hours) > 0 else None,
            'intervalo_mediano_h': intervals_hours.median() if len(intervals_hours) > 0 else None,
            'intervalo_std_h': intervals_hours.std() if len(intervals_hours) > 0 else None,
            'intervalo_min_h': intervals_hours.min() if len(intervals_hours) > 0 else None,
            'intervalo_max_h': intervals_hours.max() if len(intervals_hours) > 0 else None,
            'hora_pico': df_alert['hour'].mode().iloc[0] if len(df_alert['hour'].mode()) > 0 else 12,
            'pct_fins_semana': df_alert['is_weekend'].mean() * 100,
            'pct_horario_comercial': df_alert['is_business_hours'].mean() * 100,
            'variabilidade_intervalo': intervals_hours.std() / intervals_hours.mean() 
                                      if len(intervals_hours) > 0 and intervals_hours.mean() > 0 else 0,
            'primeiro_alerta': df_alert['created_on'].min(),
            'ultimo_alerta': df_alert['created_on'].max()
        }
        return metrics
    except Exception:
        return None


def process_alert_chunk(alert_ids, df_original, max_gap_hours=24, min_group_size=3, spike_threshold_multiplier=5):
    """Processa chunk de alertas"""
    return [metrics for alert_id in alert_ids 
            if (metrics := process_single_alert(alert_id, df_original, max_gap_hours, min_group_size, spike_threshold_multiplier))]


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class StreamlitAlertAnalyzer:
    def __init__(self):
        self.df_original = None
        self.df_all_alerts = None
        self.df = None
        self.dates = None
        self.alert_id = None
        self.max_gap_hours = 24
        self.min_group_size = 3
        self.spike_threshold_multiplier = 5

    def load_data(self, uploaded_file):
        """Carrega dados do CSV"""
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado: {len(df_raw)} registros")
            
            with st.expander("📋 Preview dos Dados"):
                st.write(f"**Colunas:** {list(df_raw.columns)}")
                st.dataframe(df_raw.head())
            
            if 'created_on' not in df_raw.columns or 'short_ci' not in df_raw.columns:
                st.error("❌ Colunas obrigatórias: 'created_on' e 'short_ci'")
                return False
            
            df_raw['created_on'] = pd.to_datetime(df_raw['created_on'])
            df_raw = df_raw.dropna(subset=['created_on'])
            df_raw = df_raw.sort_values(['short_ci', 'created_on']).reset_index(drop=True)
            
            self.df_original = df_raw
            st.sidebar.write(f"**IDs disponíveis:** {len(df_raw['short_ci'].unique())}")
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

        df_filtered, groups_info = identify_alert_groups(
            df_filtered, self.max_gap_hours, self.min_group_size, self.spike_threshold_multiplier
        )

        self.df = df_filtered
        self.dates = df_filtered['created_on']
        self.alert_id = alert_id
        self.groups_info = groups_info
        return True

    def prepare_global_analysis(self, use_multiprocessing=True, max_gap_hours=24, 
                               min_group_size=3, spike_threshold_multiplier=5):
        """Prepara análise global COM MULTIPROCESSING"""
        st.header("🌍 Análise Global")
        
        self.df = self.df_original.copy()
        self.max_gap_hours = max_gap_hours
        self.min_group_size = min_group_size
        self.spike_threshold_multiplier = spike_threshold_multiplier
        
        unique_ids = self.df_original['short_ci'].unique()
        total_ids = len(unique_ids)
        st.info(f"📊 Processando {total_ids} Alert IDs...")
        
        alert_metrics = []
        
        if use_multiprocessing and total_ids > 10:  # Só usar MP se valer a pena
            n_processes = min(cpu_count(), total_ids, 8)  # Máximo 8 processos
            st.write(f"🚀 Usando {n_processes} processos paralelos")
            
            chunk_size = max(1, total_ids // n_processes)
            id_chunks = [unique_ids[i:i + chunk_size] for i in range(0, total_ids, chunk_size)]
            
            progress_bar = st.progress(0)
            
            process_func = partial(
                process_alert_chunk,
                df_original=self.df_original,
                max_gap_hours=max_gap_hours,
                min_group_size=min_group_size,
                spike_threshold_multiplier=spike_threshold_multiplier
            )
            
            try:
                with Pool(processes=n_processes) as pool:
                    results = pool.map(process_func, id_chunks)
                    for result in results:
                        alert_metrics.extend(result)
                progress_bar.progress(1.0)
                st.success(f"✅ {len(alert_metrics)} alertas processados")
            except Exception as e:
                st.error(f"❌ Erro no multiprocessing: {e}")
                use_multiprocessing = False
                alert_metrics = []
        
        if not use_multiprocessing or len(alert_metrics) == 0:
            progress_bar = st.progress(0)
            for i, alert_id in enumerate(unique_ids):
                progress_bar.progress((i + 1) / total_ids)
                metrics = process_single_alert(
                    alert_id, self.df_original, 
                    max_gap_hours, min_group_size, spike_threshold_multiplier
                )
                if metrics:
                    alert_metrics.append(metrics)
            progress_bar.empty()
        
        self.df_all_alerts = pd.DataFrame(alert_metrics)
        return len(self.df_all_alerts) > 0

    def batch_analyze_all_short_ci_with_multiprocessing(self, progress_bar=None):
        """
        NOVA VERSÃO COM MULTIPROCESSING para análise de reincidência em lote
        """
        try:
            if self.df_original is None or len(self.df_original) == 0:
                st.error("❌ Dados não carregados")
                return None
            
            short_ci_list = self.df_original['short_ci'].unique()
            total = len(short_ci_list)
            
            # Decidir se usa multiprocessing
            use_mp = total > 20  # Só compensa para muitos alertas
            
            if use_mp:
                n_processes = min(cpu_count(), total, 8)
                st.info(f"🚀 Usando {n_processes} processos paralelos para {total} alertas")
                
                # Dividir em chunks
                chunk_size = max(1, total // n_processes)
                chunks = [short_ci_list[i:i + chunk_size] for i in range(0, total, chunk_size)]
                
                # Processar em paralelo
                process_func = partial(analyze_chunk_recurrence, df_original=self.df_original)
                
                try:
                    all_results = []
                    with Pool(processes=n_processes) as pool:
                        for idx, chunk_results in enumerate(pool.imap(process_func, chunks)):
                            all_results.extend(chunk_results)
                            if progress_bar:
                                progress = (len(all_results) / total)
                                progress_bar.progress(progress, text=f"Processando: {len(all_results)}/{total}")
                    
                    return pd.DataFrame(all_results)
                
                except Exception as e:
                    st.warning(f"⚠️ Erro no multiprocessing: {e}. Usando modo sequencial...")
                    use_mp = False
            
            # Modo sequencial (fallback ou para poucos alertas)
            if not use_mp:
                all_results = []
                for idx, short_ci in enumerate(short_ci_list):
                    if progress_bar:
                        progress_bar.progress((idx + 1) / total, text=f"Analisando {idx + 1}/{total}: {short_ci}")
                    
                    result = analyze_single_short_ci_recurrence(short_ci, self.df_original)
                    if result:
                        all_results.append(result)
                
                return pd.DataFrame(all_results)
        
        except Exception as e:
            st.error(f"Erro: {e}")
            return None

    def complete_analysis_all_short_ci(self, progress_bar=None):
        """
        Análise COMPLETA COM MULTIPROCESSING
        Retorna DataFrame consolidado
        """
        try:
            if self.df_original is None or len(self.df_original) == 0:
                st.error("❌ Dados não carregados")
                return None
            
            # 1. Análise global (20% do progresso)
            if progress_bar:
                progress_bar.progress(0.05, text="Executando análise global...")
            
            alert_ids = self.df_original['short_ci'].unique()
            
            # Usar multiprocessing para análise global
            use_mp_global = len(alert_ids) > 20
            
            if use_mp_global:
                n_processes = min(cpu_count(), len(alert_ids), 8)
                chunk_size = max(1, len(alert_ids) // n_processes)
                chunks = [alert_ids[i:i + chunk_size] for i in range(0, len(alert_ids), chunk_size)]
                
                process_func = partial(
                    process_alert_chunk,
                    df_original=self.df_original,
                    max_gap_hours=self.max_gap_hours,
                    min_group_size=self.min_group_size,
                    spike_threshold_multiplier=self.spike_threshold_multiplier
                )
                
                results_global = []
                try:
                    with Pool(processes=n_processes) as pool:
                        for chunk_result in pool.map(process_func, chunks):
                            results_global.extend(chunk_result)
                except:
                    # Fallback sequencial
                    for alert_id in alert_ids:
                        metrics = process_single_alert(
                            alert_id, self.df_original,
                            self.max_gap_hours, self.min_group_size, self.spike_threshold_multiplier
                        )
                        if metrics:
                            results_global.append(metrics)
            else:
                # Sequencial para poucos alertas
                results_global = []
                for alert_id in alert_ids:
                    metrics = process_single_alert(
                        alert_id, self.df_original,
                        self.max_gap_hours, self.min_group_size, self.spike_threshold_multiplier
                    )
                    if metrics:
                        results_global.append(metrics)
            
            df_global = pd.DataFrame(results_global)
            df_global = df_global.rename(columns={'alert_id': 'short_ci'})
            
            if progress_bar:
                progress_bar.progress(0.20, text="Análise global concluída!")
            
            # 2. Análise de reincidência COM MULTIPROCESSING (80% do progresso)
            if progress_bar:
                progress_bar.progress(0.25, text="Iniciando análise de reincidência...")
            
            # Criar sub-progress para análise de reincidência
            df_reincidencia = self.batch_analyze_all_short_ci_with_multiprocessing(progress_bar)
            
            if df_reincidencia is None or len(df_reincidencia) == 0:
                st.error("❌ Erro na análise de reincidência")
                return None
            
            # 3. Merge dos resultados
            if progress_bar:
                progress_bar.progress(0.95, text="Consolidando resultados...")
            
            df_consolidated = pd.merge(
                df_global,
                df_reincidencia,
                on='short_ci',
                how='outer'
            )
            
            # Reordenar colunas
            priority_columns = [
                'short_ci',
                'score',  # Renomear de reincidencia_score
                'classification',  # Renomear de reincidencia_status
                'pattern_type',
                'total_ocorrencias',
                'num_grupos',
                'alertas_isolados',
                'alertas_agrupados'
            ]
            
            # Renomear colunas para simplificar
            df_consolidated = df_consolidated.rename(columns={
                'total_occurrences': 'total_ocorrencias_reinc',
                'mean_interval_hours': 'intervalo_medio_reinc'
            })
            
            # Adicionar colunas restantes
            other_columns = [col for col in df_consolidated.columns if col not in priority_columns]
            final_columns = [col for col in priority_columns + other_columns if col in df_consolidated.columns]
            
            df_consolidated = df_consolidated[final_columns]
            df_consolidated = df_consolidated.sort_values('score', ascending=False)
            
            if progress_bar:
                progress_bar.progress(1.0, text="✅ Análise completa!")
            
            return df_consolidated
        
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
        col2.metric("📅 Período (dias)", period_days)
        col3.metric("📆 Dias Únicos", unique_days)
        col4.metric("📈 Média/dia", f"{avg_per_day:.2f}")
        col5.metric("🕐 Último", self.dates.max().strftime("%d/%m %H:%M"))
        
        if unique_days == 1:
            st.warning("⚠️ Todos os alertas em apenas 1 dia - classificado como ISOLADO")
        
        # Médias de frequência
        st.markdown("---")
        st.subheader("📊 Frequências")
        
        total_hours = period_days * 24
        avg_per_hour = total / total_hours if total_hours > 0 else 0
        avg_per_week = total / (period_days / 7) if period_days > 0 else 0
        avg_per_month = total / (period_days / 30.44) if period_days > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📅 Por Dia", f"{avg_per_day:.2f}")
        col2.metric("🕐 Por Hora", f"{avg_per_hour:.4f}")
        col3.metric("📆 Por Semana", f"{avg_per_week:.2f}")
        col4.metric("📊 Por Mês", f"{avg_per_month:.2f}")
        
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
    st.title("🚨 Analisador de Alertas - Otimizado")
    st.markdown("### Com multiprocessing e critérios revisados")
    
    st.sidebar.header("⚙️ Configurações")
    
    with st.sidebar.expander("🎛️ Parâmetros", expanded=False):
        max_gap_hours = st.slider("Gap Máximo (h)", 1, 72, 24)
        min_group_size = st.slider("Tamanho Mínimo Grupo", 2, 10, 3)
        spike_threshold = st.slider("Multiplicador Spike", 2.0, 10.0, 5.0, 0.5)
    
    analysis_mode = st.sidebar.selectbox(
        "🎯 Modo",
        ["🌍 Global", "🔍 Individual", "🔄 Reincidência Global", "📊 Completa + CSV"]
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
                    analyzer.max_gap_hours = max_gap_hours
                    analyzer.min_group_size = min_group_size
                    analyzer.spike_threshold_multiplier = spike_threshold
                    
                    if analyzer.prepare_individual_analysis(selected_id):
                        st.success(f"Analisando: {selected_id}")
                        
                        tab1, tab2 = st.tabs(["📊 Básico", "🔄 Reincidência"])
                        
                        with tab1:
                            analyzer.show_basic_stats()
                        
                        with tab2:
                            recurrence_analyzer = AdvancedRecurrenceAnalyzer(analyzer.df, selected_id)
                            recurrence_analyzer.analyze()
            
            elif analysis_mode == "🔄 Reincidência Global":
                st.subheader("🔄 Análise de Reincidência Global (COM MULTIPROCESSING)")
                
                if st.sidebar.button("🚀 Executar", type="primary"):
                    if analyzer.prepare_global_analysis():
                        num_ci = len(analyzer.df['short_ci'].unique())
                        st.info(f"📊 Analisando {num_ci} Short CIs com multiprocessing...")
                        
                        progress_bar = st.progress(0)
                        
                        # USAR NOVA VERSÃO COM MULTIPROCESSING
                        results_df = analyzer.batch_analyze_all_short_ci_with_multiprocessing(progress_bar)
                        
                        progress_bar.empty()
                        
                        if results_df is not None and len(results_df) > 0:
                            st.success(f"✅ {len(results_df)} alertas analisados!")
                            
                            # Stats
                            st.subheader("📊 Resumo")
                            
                            critical = len(results_df[results_df['classification'].str.contains('CRÍTICO', na=False)])
                            high = len(results_df[results_df['classification'].str.contains('PARCIALMENTE', na=False)])
                            medium = len(results_df[results_df['classification'].str.contains('DETECTÁVEL', na=False)])
                            low = len(results_df[results_df['classification'].str.contains('NÃO', na=False)])
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("🔴 P1", critical)
                            col2.metric("🟠 P2", high)
                            col3.metric("🟡 P3", medium)
                            col4.metric("🟢 P4", low)
                            
                            # Top 20
                            st.subheader("🏆 Top 20")
                            top_20 = results_df.nlargest(20, 'score')[[
                                'short_ci', 'total_occurrences', 'score', 'classification',
                                'cv', 'regularity_score', 'predictability_score'
                            ]].round(2)
                            st.dataframe(top_20, use_container_width=True)
                            
                            # Download
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "⬇️ Download CSV",
                                csv,
                                f"reincidencia_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
            
            elif analysis_mode == "📊 Completa + CSV":
                st.subheader("📊 Análise Completa COM MULTIPROCESSING")
                
                if st.sidebar.button("🚀 Executar", type="primary"):
                    if analyzer.prepare_global_analysis():
                        st.info("⏱️ Processando com multiprocessing...")
                        
                        progress_bar = st.progress(0)
                        
                        # USAR VERSÃO COM MULTIPROCESSING
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
                            st.subheader("🏆 Top 20 Mais Críticos")
                            top_20 = df_consolidated.nlargest(20, 'score')[[
                                'short_ci', 'score', 'classification', 'pattern_type',
                                'total_ocorrencias', 'freq_dia'
                            ]].round(2)
                            st.dataframe(top_20, use_container_width=True)
                            
                            # Downloads
                            st.markdown("---")
                            st.subheader("📥 Exportar")
                            
                            col1, col2 = st.columns(2)
                            
                            # CSV Completo
                            csv_full = df_consolidated.to_csv(index=False)
                            col1.download_button(
                                "⬇️ CSV Completo",
                                csv_full,
                                f"completo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                            
                            # CSV Resumido
                            summary = df_consolidated[['short_ci', 'score', 'classification']].copy()
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


if __name__ == "__main__":
    main()
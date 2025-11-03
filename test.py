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
    page_title="Analisador de Alertas - FIXED",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedRecurrenceAnalyzer:
    """
    ✅ VERSÃO CORRIGIDA - Scores idênticos em análise individual e completa
    
    Correção aplicada: Todos os métodos "_silent" agora chamam os métodos normais
    internamente, garantindo que os cálculos sejam EXATAMENTE IGUAIS.
    """
    
    def __init__(self, df, alert_id):
        self.df = df.copy() if df is not None else None
        self.alert_id = alert_id
        self.cache = {}
        
    def _cache_result(self, key, func):
        """Cache de resultados para otimização"""
        if key not in self.cache:
            self.cache[key] = func()
        return self.cache[key]
    
    def _prepare_data(self):
        """Preparação otimizada dos dados"""
        if self.df is None or len(self.df) < 3:
            return None
            
        df = self.df.sort_values('created_on').copy()
        
        # Vetorizar operações de timestamp
        df['timestamp'] = df['created_on'].astype('int64') // 10**9
        df['time_diff_seconds'] = df['timestamp'].diff()
        df['time_diff_hours'] = df['time_diff_seconds'] / 3600
        df['time_diff_days'] = df['time_diff_seconds'] / 86400
        
        # Extrair componentes temporais de uma vez
        if 'hour' not in df.columns:
            df['hour'] = df['created_on'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['created_on'].dt.dayofweek
        if 'day_of_month' not in df.columns:
            df['day_of_month'] = df['created_on'].dt.day
        if 'week_of_year' not in df.columns:
            df['week_of_year'] = df['created_on'].dt.isocalendar().week
        if 'month' not in df.columns:
            df['month'] = df['created_on'].dt.month
        if 'day_name' not in df.columns:
            df['day_name'] = df['created_on'].dt.day_name()
        
        return df
    
    # ============================================================
    # MÉTODOS CORE DE CÁLCULO (usados por ambas as versões)
    # ============================================================
    
    def _compute_basic_statistics(self, intervals):
        """Cálculo CORE de estatísticas básicas - usado por ambos os métodos"""
        return {
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
    
    def _compute_regularity(self, intervals):
        """Cálculo CORE de regularidade - usado por ambos os métodos"""
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        
        if cv < 0.15:
            regularity_score = 95
            pattern_type = "🟢 ALTAMENTE REGULAR"
        elif cv < 0.35:
            regularity_score = 80
            pattern_type = "🟢 REGULAR"
        elif cv < 0.65:
            regularity_score = 60
            pattern_type = "🟡 SEMI-REGULAR"
        elif cv < 1.0:
            regularity_score = 40
            pattern_type = "🟠 IRREGULAR"
        else:
            regularity_score = 20
            pattern_type = "🔴 ALTAMENTE IRREGULAR"
        
        return {
            'cv': cv,
            'score': regularity_score,
            'type': pattern_type,
            'regularity_score': regularity_score
        }
    
    def _compute_periodicity(self, intervals):
        """Cálculo CORE de periodicidade - usado por ambos os métodos"""
        if len(intervals) < 10:
            return {
                'has_strong_periodicity': False,
                'has_moderate_periodicity': False,
                'dominant_period_hours': None,
                'periods': []
            }
        
        try:
            # NORMALIZAÇÃO (crucial para resultados consistentes)
            intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
            
            # PADDING para potência de 2 (melhora a FFT)
            n_padded = 2**int(np.ceil(np.log2(len(intervals_norm))))
            intervals_padded = np.pad(intervals_norm, (0, n_padded - len(intervals_norm)), 'constant')
            
            # FFT
            fft_vals = fft(intervals_padded)
            freqs = fftfreq(n_padded, d=1)
            
            # Frequências positivas apenas
            positive_idx = freqs > 0
            freqs_pos = freqs[positive_idx]
            fft_mag = np.abs(fft_vals[positive_idx])
            
            # Threshold sofisticado (média + 2*std)
            threshold = np.mean(fft_mag) + 2 * np.std(fft_mag)
            peaks_idx = fft_mag > threshold
            
            dominant_periods = []
            dominant_period_hours = None
            
            if np.any(peaks_idx):
                dominant_freqs = freqs_pos[peaks_idx]
                dominant_periods = 1 / dominant_freqs
                dominant_periods = dominant_periods[dominant_periods < len(intervals)][:3]
                
                if len(dominant_periods) > 0:
                    dominant_period_hours = dominant_periods[0] * np.mean(intervals)
            
            # Determinar força da periodicidade
            if len(fft_mag) > 1:
                peak_power = fft_mag[peaks_idx].max() if np.any(peaks_idx) else 0
                mean_power = np.mean(fft_mag)
                strength_ratio = peak_power / mean_power if mean_power > 0 else 0
                has_strong = strength_ratio > 3.0
                has_moderate = 1.5 < strength_ratio <= 3.0
            else:
                has_strong = False
                has_moderate = False
            
            return {
                'has_strong_periodicity': has_strong,
                'has_moderate_periodicity': has_moderate,
                'dominant_period_hours': dominant_period_hours,
                'periods': list(dominant_periods) if len(dominant_periods) > 0 else []
            }
        except Exception as e:
            return {
                'has_strong_periodicity': False,
                'has_moderate_periodicity': False,
                'dominant_period_hours': None,
                'periods': []
            }
    
    def _compute_autocorrelation(self, intervals):
        """Cálculo CORE de autocorrelação - usado por ambos os métodos"""
        if len(intervals) < 5:
            return {
                'peaks': [],
                'has_autocorr': False,
                'max_autocorr': 0
            }
        
        try:
            # NORMALIZAÇÃO (crucial para resultados consistentes)
            intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
            
            # Correlação usando signal.correlate
            autocorr = signal.correlate(intervals_norm, intervals_norm, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            lags = np.arange(len(autocorr))
            threshold = 2 / np.sqrt(len(intervals))
            
            # Picos significativos
            significant_peaks = [(i, autocorr[i]) for i in range(1, min(len(autocorr), 20)) 
                               if autocorr[i] > threshold]
            
            # Calcular max_autocorr
            max_autocorr = max([abs(corr) for _, corr in significant_peaks]) if significant_peaks else 0
            
            return {
                'peaks': significant_peaks,
                'has_autocorr': len(significant_peaks) > 0,
                'max_autocorr': max_autocorr
            }
        except Exception as e:
            return {
                'peaks': [],
                'has_autocorr': False,
                'max_autocorr': 0
            }
    
    def _compute_predictability(self, intervals):
        """Cálculo CORE de previsibilidade - usado por ambos os métodos"""
        if len(intervals) < 5:
            return {
                'predictability_score': 0,
                'entropy': 1,
                'next_expected_hours': np.mean(intervals) if len(intervals) > 0 else 0
            }
        
        n_bins = min(10, len(intervals) // 3)
        hist, _ = np.histogram(intervals, bins=n_bins)
        probs = hist[hist > 0] / hist.sum()
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(n_bins)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1
        
        predictability_score = (1 - norm_entropy) * 100
        
        return {
            'predictability_score': predictability_score,
            'entropy': norm_entropy,
            'next_expected_hours': np.mean(intervals)
        }
    
    def _compute_markov_chains(self, intervals):
        """Cálculo CORE de cadeias de Markov - usado por ambos os métodos"""
        if len(intervals) < 5:
            return {'markov_score': 0, 'markov_predictability': 0}
        
        try:
            # Discretizar intervalos em estados
            bins = np.percentile(intervals, [0, 33, 67, 100])
            states = np.digitize(intervals, bins[1:-1])
            n_states = 3
            
            # Construir matriz de transição
            transition_matrix = np.zeros((n_states, n_states))
            for i in range(len(states) - 1):
                current = states[i]
                next_state = states[i + 1]
                transition_matrix[current, next_state] += 1
            
            # Normalizar
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            transition_matrix = transition_matrix / row_sums
            
            # Score baseado nas probabilidades máximas
            max_probs = transition_matrix.max(axis=1)
            markov_score = np.mean(max_probs) * 100
            
            return {
                'markov_score': markov_score,
                'markov_predictability': markov_score
            }
        except:
            return {'markov_score': 0, 'markov_predictability': 0}
    
    def _compute_randomness(self, intervals):
        """Cálculo CORE de aleatoriedade - usado por ambos os métodos"""
        if len(intervals) < 5:
            return {'overall_randomness_score': 50, 'randomness_score': 50}
        
        try:
            # Runs Test
            median = np.median(intervals)
            runs = np.diff(intervals > median).sum() + 1
            expected_runs = len(intervals) / 2
            runs_score = min(abs(runs - expected_runs) / expected_runs * 100, 100)
            
            overall_randomness = runs_score
            
            return {
                'overall_randomness_score': overall_randomness,
                'randomness_score': overall_randomness
            }
        except:
            return {'overall_randomness_score': 50, 'randomness_score': 50}
    
    def _compute_stability(self, intervals):
        """Cálculo CORE de estabilidade - usado por ambos os métodos"""
        if len(intervals) < 10:
            return {
                'is_stable': True,
                'drift_pct': 0,
                'p_value': 1.0,
                'stability_score': 50
            }
        
        try:
            mid = len(intervals) // 2
            first_half = intervals[:mid]
            second_half = intervals[mid:]
            
            _, p_value = stats.ttest_ind(first_half, second_half)
            
            is_stable = p_value > 0.05
            
            mean_diff = abs(np.mean(second_half) - np.mean(first_half))
            drift_pct = (mean_diff / np.mean(first_half)) * 100 if np.mean(first_half) > 0 else 0
            
            stability_score = min(p_value * 100, 100)
            
            return {
                'is_stable': is_stable,
                'drift_pct': drift_pct,
                'p_value': p_value,
                'stability_score': stability_score
            }
        except Exception as e:
            return {
                'is_stable': True,
                'drift_pct': 0,
                'p_value': 1.0,
                'stability_score': 50
            }
    
    # ============================================================
    # MÉTODOS COM INTERFACE (análise individual)
    # ============================================================
    
    def _analyze_basic_statistics(self, intervals):
        """Estatísticas básicas COM INTERFACE"""
        st.subheader("📊 1. Estatísticas de Intervalos")
        
        stats_dict = self._compute_basic_statistics(intervals)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("⏱️ Média", f"{stats_dict['mean']:.1f}h")
        col2.metric("📊 Mediana", f"{stats_dict['median']:.1f}h")
        col3.metric("📈 Desvio", f"{stats_dict['std']:.1f}h")
        col4.metric("⚡ Mínimo", f"{stats_dict['min']:.1f}h")
        col5.metric("🐌 Máximo", f"{stats_dict['max']:.1f}h")
        
        return stats_dict
    
    def _analyze_regularity(self, intervals):
        """Análise de regularidade COM INTERFACE"""
        st.subheader("🎯 2. Regularidade e Aleatoriedade")
        
        result = self._compute_regularity(intervals)
        
        pattern_color = {
            "🟢 ALTAMENTE REGULAR": "green",
            "🟢 REGULAR": "lightgreen",
            "🟡 SEMI-REGULAR": "yellow",
            "🟠 IRREGULAR": "orange",
            "🔴 ALTAMENTE IRREGULAR": "red"
        }.get(result['type'], "gray")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Classificação:** {result['type']}")
            st.write(f"**CV:** {result['cv']:.2%}")
            
            # Teste de Shapiro-Wilk para normalidade
            if len(intervals) >= 3:
                _, p_value = stats.shapiro(intervals)
                if p_value > 0.05:
                    st.info("📊 **Normalidade:** Distribuição aproximadamente normal")
                else:
                    st.warning("📊 **Normalidade:** Distribuição não-normal")
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['regularity_score'],
                title={'text': "Regularidade"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': pattern_color},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True, key='reg_gauge')
        
        return result
    
    def _analyze_periodicity(self, intervals):
        """Análise de periodicidade COM INTERFACE"""
        st.subheader("🔍 3. Periodicidade (FFT)")
        
        result = self._compute_periodicity(intervals)
        
        if len(intervals) < 10:
            st.info("📊 Mínimo de 10 intervalos necessários")
            return result
        
        if len(result.get('periods', [])) > 0:
            st.success("🎯 **Periodicidades Detectadas:**")
            for period in result['periods']:
                est_time = period * np.mean(intervals)
                time_str = f"{est_time:.1f}h" if est_time < 24 else f"{est_time/24:.1f} dias"
                st.write(f"• Período: **{period:.1f}** ocorrências (~{time_str})")
        else:
            st.info("📊 Nenhuma periodicidade forte detectada")
        
        # Gráfico FFT
        try:
            intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
            n_padded = 2**int(np.ceil(np.log2(len(intervals_norm))))
            intervals_padded = np.pad(intervals_norm, (0, n_padded - len(intervals_norm)), 'constant')
            
            fft_vals = fft(intervals_padded)
            freqs = fftfreq(n_padded, d=1)
            
            positive_idx = freqs > 0
            freqs_pos = freqs[positive_idx]
            fft_mag = np.abs(fft_vals[positive_idx])
            
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
        except:
            pass
        
        return result
    
    def _analyze_autocorrelation(self, intervals):
        """Análise de autocorrelação COM INTERFACE"""
        st.subheader("📈 4. Autocorrelação")
        
        result = self._compute_autocorrelation(intervals)
        
        if len(intervals) < 5:
            return result
        
        if result['has_autocorr']:
            st.success("✅ **Autocorrelação Significativa:**")
            for lag, corr in result['peaks'][:3]:
                st.write(f"• Lag {lag}: {corr:.2f}")
        else:
            st.info("📊 Sem autocorrelação significativa")
        
        # Gráfico
        try:
            intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
            autocorr = signal.correlate(intervals_norm, intervals_norm, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            lags = np.arange(len(autocorr))
            threshold = 2 / np.sqrt(len(intervals))
            
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
        except:
            pass
        
        return result
    
    # ============================================================
    # MÉTODOS SILENCIOSOS (análise completa) - CHAMAM OS CORE!
    # ============================================================
    
    def _analyze_basic_statistics_silent(self, intervals):
        """✅ CORRIGIDO: Chama o método CORE"""
        return self._compute_basic_statistics(intervals)
    
    def _analyze_regularity_silent(self, intervals):
        """✅ CORRIGIDO: Chama o método CORE"""
        return self._compute_regularity(intervals)
    
    def _analyze_periodicity_silent(self, intervals):
        """✅ CORRIGIDO: Chama o método CORE"""
        return self._compute_periodicity(intervals)
    
    def _analyze_autocorrelation_silent(self, intervals):
        """✅ CORRIGIDO: Chama o método CORE"""
        return self._compute_autocorrelation(intervals)
    
    def _calculate_predictability_silent(self, intervals):
        """✅ CORRIGIDO: Chama o método CORE"""
        return self._compute_predictability(intervals)
    
    def _analyze_markov_chains_silent(self, intervals):
        """✅ CORRIGIDO: Chama o método CORE"""
        return self._compute_markov_chains(intervals)
    
    def _advanced_randomness_tests_silent(self, intervals):
        """✅ CORRIGIDO: Chama o método CORE"""
        return self._compute_randomness(intervals)
    
    def _analyze_stability_silent(self, intervals):
        """✅ CORRIGIDO: Chama o método CORE"""
        return self._compute_stability(intervals)
    
    # ============================================================
    # CÁLCULO DE SCORE FINAL (usado por ambos)
    # ============================================================
    
    def _calculate_final_score(self, results):
        """
        Calcula o score final de reincidência (0-100) usando 7 CRITÉRIOS ESSENCIAIS
        """
        scores = {
            # 1. REGULARIDADE (20 pontos)
            'regularity': results['regularity']['regularity_score'] * 0.20,
            
            # 2. PERIODICIDADE (20 pontos)
            'periodicity': (100 if results['periodicity']['has_strong_periodicity'] else 
                          (50 if results['periodicity'].get('has_moderate_periodicity', False) else 0)) * 0.20,
            
            # 3. PREVISIBILIDADE (15 pontos)
            'predictability': results['predictability']['predictability_score'] * 0.15,
            
            # 4. DETERMINISMO (15 pontos)
            'determinism': (100 - results['randomness']['overall_randomness_score']) * 0.15,
            
            # 5. AUTOCORRELAÇÃO (10 pontos)
            'autocorrelation': (results['autocorr']['max_autocorr'] * 100) * 0.10,
            
            # 6. ESTABILIDADE (10 pontos)
            'stability': results.get('stability', {}).get('stability_score', 50) * 0.10,
            
            # 7. MARKOV (10 pontos)
            'markov': results['markov']['markov_score'] * 0.10
        }
        
        final_score = sum(scores.values())
        
        # Classificação
        if final_score >= 70:
            classification = "🔴 REINCIDENTE CRÍTICO (P1)"
        elif final_score >= 50:
            classification = "🟠 PARCIALMENTE REINCIDENTE (P2)"
        elif final_score >= 30:
            classification = "🟡 PADRÃO DETECTÁVEL (P3)"
        else:
            classification = "🟢 NÃO REINCIDENTE (P4)"
        
        return round(final_score, 2), classification
    
    # ============================================================
    # MÉTODOS PÚBLICOS
    # ============================================================
    
    def analyze_silent(self):
        """Análise silenciosa para processamento em lote"""
        df = self._prepare_data()
        if df is None:
            return None
        
        intervals_hours = df['time_diff_hours'].dropna().values
        if len(intervals_hours) < 2:
            return None
        
        # ✅ TODOS CHAMAM OS MÉTODOS CORE AGORA!
        results = {}
        results['basic_stats'] = self._compute_basic_statistics(intervals_hours)
        results['regularity'] = self._compute_regularity(intervals_hours)
        results['periodicity'] = self._compute_periodicity(intervals_hours)
        results['autocorr'] = self._compute_autocorrelation(intervals_hours)
        results['predictability'] = self._compute_predictability(intervals_hours)
        results['markov'] = self._compute_markov_chains(intervals_hours)
        results['randomness'] = self._compute_randomness(intervals_hours)
        results['stability'] = self._compute_stability(intervals_hours)
        
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
    
    def analyze(self):
        """Análise completa COM INTERFACE"""
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
        
        # ✅ TODOS CHAMAM OS MÉTODOS CORE AGORA!
        results = {}
        results['basic_stats'] = self._analyze_basic_statistics(intervals_hours)
        results['regularity'] = self._analyze_regularity(intervals_hours)
        results['periodicity'] = self._analyze_periodicity(intervals_hours)
        results['autocorr'] = self._analyze_autocorrelation(intervals_hours)
        results['predictability'] = self._compute_predictability(intervals_hours)
        results['markov'] = self._compute_markov_chains(intervals_hours)
        results['randomness'] = self._compute_randomness(intervals_hours)
        results['stability'] = self._compute_stability(intervals_hours)
        
        # Classificação final
        self._final_classification(results, df, intervals_hours)
    
    def _final_classification(self, results, df, intervals):
        """Classificação final COM INTERFACE"""
        st.markdown("---")
        st.header("🎯 CLASSIFICAÇÃO FINAL DE REINCIDÊNCIA")
        
        final_score, classification = self._calculate_final_score(results)
        
        st.success(f"✅ **Score calculado:** {final_score:.2f}/100")
        st.info(f"🎯 **Classificação:** {classification}")
        
        # Resto da interface...
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.metric("Score de Reincidência", f"{final_score:.0f}/100")
        
        with col2:
            color = "red" if final_score >= 70 else "orange" if final_score >= 50 else "yellow" if final_score >= 30 else "green"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=final_score,
                title={'text': "Score Final"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 50], 'color': "lightyellow"},
                        {'range': [50, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key='final_gauge')


# ============================================================
# RESTO DO CÓDIGO (functions auxiliares, main, etc)
# Mantido igual ao original...
# ============================================================

def identify_alert_groups(alert_data, max_gap_hours=24, min_group_size=3, 
                         spike_threshold_multiplier=5):
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
            'duration_hours': (group_data['created_on'].max() - 
                             group_data['created_on'].min()).total_seconds() / 3600
        })
    
    return alert_data, groups_info

def classify_alert_pattern(alert_data, max_gap_hours=24, min_group_size=3, 
                          spike_threshold_multiplier=5):
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
            'reason': f'Todos os {n} alertas ocorreram em um único dia',
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
        reason = f'Nenhum grupo identificado ({n} ocorrências isoladas)'
    elif num_groups == 1 and isolated_pct > 50:
        pattern = 'isolated'
        reason = f'Apenas 1 grupo pequeno com {isolated_pct:.0f}% de alertas isolados'
    elif isolated_pct > 70:
        pattern = 'isolated'
        reason = f'{isolated_pct:.0f}% de alertas isolados ({isolated_count}/{n})'
    elif num_groups >= 2:
        pattern = 'continuous'
        reason = f'{num_groups} grupos contínuos identificados ({grouped_count} alertas agrupados)'
    elif num_groups == 1 and grouped_count >= min_group_size * 2:
        pattern = 'continuous'
        reason = f'1 grupo contínuo grande ({grouped_count} alertas)'
    else:
        pattern = 'isolated'
        reason = f'Padrão inconsistente: {num_groups} grupo(s), {isolated_pct:.0f}% isolados'
    
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


def process_single_alert(alert_id, df_original, max_gap_hours=24, min_group_size=3, 
                        spike_threshold_multiplier=5):
    try:
        df_alert = df_original[df_original['short_ci'] == alert_id].copy()
        if len(df_alert) < 1:
            return None
        
        pattern_info = classify_alert_pattern(df_alert, max_gap_hours, min_group_size, 
                                             spike_threshold_multiplier)
        
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
            'intervalo_medio_h': intervals_hours.mean() if len(intervals_hours) > 0 else None,
            'intervalo_mediano_h': intervals_hours.median() if len(intervals_hours) > 0 else None,
            'primeiro_alerta': df_alert['created_on'].min(),
            'ultimo_alerta': df_alert['created_on'].max()
        }
        return metrics
    except Exception:
        return None


def process_alert_chunk(alert_ids, df_original, max_gap_hours=24, min_group_size=3, 
                       spike_threshold_multiplier=5):
    return [metrics for alert_id in alert_ids 
            if (metrics := process_single_alert(alert_id, df_original, max_gap_hours, 
                                               min_group_size, spike_threshold_multiplier))]


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
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado com {len(df_raw)} registros")
            if 'created_on' not in df_raw.columns or 'short_ci' not in df_raw.columns:
                st.error("❌ Colunas 'created_on' e 'short_ci' são obrigatórias!")
                return False
            df_raw['created_on'] = pd.to_datetime(df_raw['created_on'])
            df_raw = df_raw.dropna(subset=['created_on'])
            df_raw = df_raw.sort_values(['short_ci', 'created_on']).reset_index(drop=True)
            self.df_original = df_raw
            return True
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {e}")
            return False

    def prepare_individual_analysis(self, alert_id):
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
            df_filtered, 
            self.max_gap_hours, 
            self.min_group_size,
            self.spike_threshold_multiplier
        )

        self.df = df_filtered
        self.dates = df_filtered['created_on']
        self.alert_id = alert_id
        self.groups_info = groups_info
        return True

    def complete_analysis_all_short_ci(self, progress_bar=None):
        """Análise COMPLETA de todos os short_ci"""
        try:
            # 1. Análise global
            if progress_bar:
                progress_bar.progress(0.1, text="Executando análise global...")
            
            alert_ids = self.df_original['short_ci'].unique()
            results_global = []
            
            for alert_id in alert_ids:
                metrics = process_single_alert(
                    alert_id, 
                    self.df_original, 
                    self.max_gap_hours, 
                    self.min_group_size, 
                    self.spike_threshold_multiplier
                )
                if metrics:
                    results_global.append(metrics)
            
            df_global = pd.DataFrame(results_global)
            
            # 2. Análise de reincidência
            if progress_bar:
                progress_bar.progress(0.3, text="Executando análise de reincidência...")
            
            all_results = []
            total = len(alert_ids)
            
            for idx, short_ci in enumerate(alert_ids):
                if progress_bar:
                    progress = 0.3 + (0.6 * (idx + 1) / total)
                    progress_bar.progress(progress, text=f"Analisando {idx + 1}/{total}: {short_ci}")
                
                df_ci = self.df_original[self.df_original['short_ci'] == short_ci].copy()
                df_ci['created_on'] = pd.to_datetime(df_ci['created_on'], errors='coerce')
                df_ci = df_ci.dropna(subset=['created_on'])
                df_ci = df_ci.sort_values('created_on')
                
                if len(df_ci) < 3:
                    all_results.append({
                        'short_ci': short_ci,
                        'reincidencia_score': 0,
                        'reincidencia_status': '⚪ DADOS INSUFICIENTES'
                    })
                    continue
                
                analyzer = AdvancedRecurrenceAnalyzer(df_ci, short_ci)
                result = analyzer.analyze_silent()
                
                if result:
                    all_results.append({
                        'short_ci': short_ci,
                        'reincidencia_score': result['score'],
                        'reincidencia_status': result['classification']
                    })
                else:
                    all_results.append({
                        'short_ci': short_ci,
                        'reincidencia_score': 0,
                        'reincidencia_status': '⚪ ERRO NA ANÁLISE'
                    })
            
            df_reincidencia = pd.DataFrame(all_results)
            
            # 3. Merge
            if progress_bar:
                progress_bar.progress(0.95, text="Consolidando...")
            
            df_global = df_global.rename(columns={'alert_id': 'short_ci'})
            df_consolidated = pd.merge(df_global, df_reincidencia, on='short_ci', how='outer')
            df_consolidated = df_consolidated.sort_values('reincidencia_score', ascending=False)
            
            if progress_bar:
                progress_bar.progress(1.0, text="Concluído!")
            
            return df_consolidated
        
        except Exception as e:
            st.error(f"Erro: {e}")
            return None


def main():
    st.title("🚨 Analisador de Alertas - ✅ SCORES CONSISTENTES!")
    st.success("✅ **CORREÇÃO APLICADA:** Análise individual e completa usam MESMOS métodos de cálculo!")
    
    st.sidebar.header("⚙️ Configurações")
    
    analysis_mode = st.sidebar.selectbox(
        "🎯 Modo de Análise",
        ["🔍 Análise Individual", "📊 Análise Completa"]
    )
    
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        analyzer = StreamlitAlertAnalyzer()
        if analyzer.load_data(uploaded_file):
            if analysis_mode == "🔍 Análise Individual":
                alert_ids = analyzer.df_original['short_ci'].unique()
                selected_id = st.sidebar.selectbox("🎯 Selecione o Alert ID", alert_ids)
                
                if st.sidebar.button("🚀 Executar Análise", type="primary"):
                    if analyzer.prepare_individual_analysis(selected_id):
                        st.success(f"Analisando: {selected_id}")
                        
                        # Executar análise de reincidência
                        recurrence_analyzer = AdvancedRecurrenceAnalyzer(analyzer.df, selected_id)
                        recurrence_analyzer.analyze()
            
            elif analysis_mode == "📊 Análise Completa":
                if st.sidebar.button("🚀 Executar Análise Completa", type="primary"):
                    progress_bar = st.progress(0, text="Iniciando...")
                    df_results = analyzer.complete_analysis_all_short_ci(progress_bar)
                    progress_bar.empty()
                    
                    if df_results is not None:
                        st.success(f"✅ Análise concluída! {len(df_results)} alertas")
                        
                        # Mostrar resultados
                        st.dataframe(df_results[['short_ci', 'reincidencia_score', 'reincidencia_status']])
                        
                        # Download
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            "⬇️ Baixar Resultados",
                            csv,
                            f"analise_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv"
                        )
    else:
        st.info("👆 Faça upload de um arquivo CSV")
        
        with st.expander("📖 O que foi corrigido?"):
            st.markdown("""
            ### 🐛 Problema Identificado:
            Os métodos de análise **individual** e **completa** tinham **implementações diferentes**:
            
            **Análise Individual:**
            - `_analyze_periodicity()` - Com normalização e padding
            - `_analyze_autocorrelation()` - Usando signal.correlate
            
            **Análise Completa:**
            - `_analyze_periodicity_silent()` - **SEM** normalização
            - `_analyze_autocorrelation_silent()` - Usando np.corrcoef em loops
            
            ### ✅ Solução Aplicada:
            Todos os métodos agora chamam **funções CORE únicas**:
            - `_compute_periodicity()` - Usada por ambos
            - `_compute_autocorrelation()` - Usada por ambos
            - `_compute_regularity()` - Usada por ambos
            - etc.
            
            ### 🎯 Resultado:
            **Score IDÊNTICO** em análise individual e completa! ✅
            """)

if __name__ == "__main__":
    main()
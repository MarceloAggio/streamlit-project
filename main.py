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
from functools import partial
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Analisador de Alertas - Completo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CLASSE DE ANÁLISE AVANÇADA DE RECORRÊNCIA
# ============================================================

class AdvancedRecurrenceAnalyzer:
    """Analisador avançado de padrões de reincidência com otimizações"""
    
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
    
    def analyze(self):
        """Método principal de análise"""
        st.header("🔄 Análise Avançada de Reincidência Temporal")
        
        df = self._prepare_data()
        if df is None:
            st.warning("⚠️ Dados insuficientes (mínimo 3 ocorrências).")
            return
        
        st.info(f"📊 Analisando **{len(df)}** ocorrências do Alert ID: **{self.alert_id}**")
        
        intervals_hours = df['time_diff_hours'].dropna().values
        if len(intervals_hours) < 2:
            st.warning("⚠️ Intervalos insuficientes.")
            return
        
        # Executar todas as análises
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
        results['trend'] = self._analyze_trend(df)
        results['predictability'] = self._calculate_predictability(intervals_hours)
        results['stability'] = self._analyze_stability(intervals_hours, df)
        results['pattern_mining'] = self._mine_patterns(intervals_hours)
        
        # Classificação final consolidada
        self._final_classification(results, df, intervals_hours)
    
    def _analyze_basic_statistics(self, intervals):
        """Estatísticas básicas otimizadas"""
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
        """Análise de regularidade com testes estatísticos"""
        st.subheader("🎯 2. Regularidade e Aleatoriedade")
        
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
        
        if cv < 0.15:
            pattern_type = "🟢 ALTAMENTE REGULAR"
            pattern_color = "green"
            regularity_score = 95
        elif cv < 0.35:
            pattern_type = "🟢 REGULAR"
            pattern_color = "lightgreen"
            regularity_score = 80
        elif cv < 0.65:
            pattern_type = "🟡 SEMI-REGULAR"
            pattern_color = "yellow"
            regularity_score = 60
        elif cv < 1.0:
            pattern_type = "🟠 IRREGULAR"
            pattern_color = "orange"
            regularity_score = 40
        else:
            pattern_type = "🔴 ALTAMENTE IRREGULAR"
            pattern_color = "red"
            regularity_score = 20
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Classificação:** {pattern_type}")
            st.write(f"**CV:** {cv:.2%}")
            
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
                value=regularity_score,
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
        
        return {'cv': cv, 'score': regularity_score, 'type': pattern_type}
    
    def _analyze_periodicity(self, intervals):
        """Análise de periodicidade com FFT"""
        st.subheader("🔍 3. Periodicidade (FFT)")
        
        if len(intervals) < 10:
            st.info("📊 Mínimo de 10 intervalos necessários")
            return {}
        
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
        if np.any(peaks_idx):
            dominant_freqs = freqs_pos[peaks_idx]
            dominant_periods = 1 / dominant_freqs
            dominant_periods = dominant_periods[dominant_periods < len(intervals)][:3]
            
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
        
        return {'periods': dominant_periods, 'has_periodicity': len(dominant_periods) > 0}
    
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
        
        return {'peaks': significant_peaks, 'has_autocorr': len(significant_peaks) > 0}
    
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
            st.warning(f"⚠️ **{anomaly_rate:.1f}%** de anomalias detectadas")
        else:
            st.success("✅ Baixa taxa de anomalias")
        
        return {'anomaly_rate': anomaly_rate, 'total_anomalies': total_anomalies}
    
    def _analyze_trend(self, df):
        """Análise de tendência temporal"""
        df_sorted = df.sort_values('created_on')
        
        df_sorted['week'] = df_sorted['created_on'].dt.to_period('W')
        weekly_counts = df_sorted.groupby('week').size()
        
        if len(weekly_counts) < 3:
            return {'has_trend': False}
        
        x = np.arange(len(weekly_counts))
        y = weekly_counts.values
        
        slope, _, _, p_value, _ = stats.linregress(x, y)
        
        has_trend = p_value < 0.05
        trend_type = 'increasing' if slope > 0 else 'decreasing'
        
        return {
            'has_trend': has_trend,
            'trend_type': trend_type if has_trend else 'stable',
            'slope': slope,
            'p_value': p_value
        }
    
    def _calculate_predictability(self, intervals):
        """Calcula score de previsibilidade"""
        if len(intervals) < 5:
            return {'score': 0}
        
        n_bins = min(10, len(intervals) // 3)
        hist, _ = np.histogram(intervals, bins=n_bins)
        probs = hist[hist > 0] / hist.sum()
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(n_bins)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1
        
        predictability_score = (1 - norm_entropy) * 100
        
        return {'score': predictability_score, 'entropy': norm_entropy}
    
    def _analyze_stability(self, intervals, df):
        """Análise de estabilidade do padrão"""
        if len(intervals) < 10:
            return {'is_stable': True}
        
        mid = len(intervals) // 2
        first_half = intervals[:mid]
        second_half = intervals[mid:]
        
        _, p_value = stats.ttest_ind(first_half, second_half)
        
        is_stable = p_value > 0.05
        
        mean_diff = abs(np.mean(second_half) - np.mean(first_half))
        drift_pct = (mean_diff / np.mean(first_half)) * 100 if np.mean(first_half) > 0 else 0
        
        return {
            'is_stable': is_stable,
            'drift_pct': drift_pct,
            'p_value': p_value
        }
    
    def _mine_patterns(self, intervals):
        """Mining de padrões sequenciais"""
        if len(intervals) < 10:
            return {}
        
        q1, q2, q3 = np.percentile(intervals, [25, 50, 75])
        
        def categorize(val):
            if val <= q1:
                return 'A'
            elif val <= q2:
                return 'B'
            elif val <= q3:
                return 'C'
            else:
                return 'D'
        
        sequence = ''.join([categorize(i) for i in intervals])
        
        patterns = defaultdict(int)
        for n in [2, 3]:
            for i in range(len(sequence) - n + 1):
                ngram = sequence[i:i+n]
                patterns[ngram] += 1
        
        frequent = {k: v for k, v in patterns.items() if v >= 3}
        
        return {'frequent_patterns': frequent, 'has_patterns': len(frequent) > 0}
    
    def _final_classification(self, results, df, intervals):
        """Classificação final consolidada"""
        st.markdown("---")
        st.header("🎯 CLASSIFICAÇÃO FINAL DE REINCIDÊNCIA")
        
        score = 0
        max_score = 100
        criteria = []
        
        # 1. Regularidade (20 pontos)
        cv = results['basic_stats']['cv']
        if cv < 0.35:
            score += 20
            criteria.append(("✅ Alta regularidade", 20))
        elif cv < 0.65:
            score += 12
            criteria.append(("🟡 Regularidade moderada", 12))
        else:
            criteria.append(("❌ Baixa regularidade", 0))
        
        # 2. Periodicidade (15 pontos)
        if results.get('periodicity', {}).get('has_periodicity', False):
            score += 15
            criteria.append(("✅ Periodicidade detectada", 15))
        else:
            criteria.append(("❌ Sem periodicidade", 0))
        
        # 3. Autocorrelação (15 pontos)
        if results.get('autocorr', {}).get('has_autocorr', False):
            score += 15
            criteria.append(("✅ Autocorrelação significativa", 15))
        else:
            criteria.append(("❌ Sem autocorrelação", 0))
        
        # 4. Concentração temporal (15 pontos)
        hourly_conc = results.get('temporal', {}).get('hourly_concentration', 0)
        daily_conc = results.get('temporal', {}).get('daily_concentration', 0)
        if hourly_conc > 50 or daily_conc > 50:
            score += 15
            criteria.append(("✅ Alta concentração temporal", 15))
        else:
            criteria.append(("❌ Distribuição uniforme", 0))
        
        # 5. Clusters (10 pontos)
        if results.get('clusters', {}).get('n_clusters', 0) > 1:
            score += 10
            criteria.append(("✅ Clusters identificados", 10))
        else:
            criteria.append(("❌ Sem clusters", 0))
        
        # 6. Previsibilidade (10 pontos)
        pred_score = results.get('predictability', {}).get('score', 0)
        if pred_score > 60:
            score += 10
            criteria.append(("✅ Alta previsibilidade", 10))
        else:
            criteria.append(("❌ Baixa previsibilidade", 0))
        
        # 7. Estabilidade (10 pontos)
        if results.get('stability', {}).get('is_stable', True):
            score += 10
            criteria.append(("✅ Padrão estável", 10))
        else:
            criteria.append(("⚠️ Padrão instável", 0))
        
        # 8. Ausência de bursts (5 pontos)
        if not results.get('bursts', {}).get('has_bursts', False):
            score += 5
            criteria.append(("✅ Sem rajadas", 5))
        else:
            criteria.append(("⚠️ Presença de bursts", 0))
        
        # Determinar classificação
        if score >= 70:
            classification = "🔴 ALERTA REINCIDENTE"
            level = "CRÍTICO"
            color = "red"
            recommendation = "**Ação Imediata:** Criar automação, runbook e investigar causa raiz"
            priority = "P1"
        elif score >= 50:
            classification = "🟠 PARCIALMENTE REINCIDENTE"
            level = "ALTO"
            color = "orange"
            recommendation = "**Ação Recomendada:** Monitorar evolução e considerar automação"
            priority = "P2"
        elif score >= 30:
            classification = "🟡 PADRÃO DETECTÁVEL"
            level = "MÉDIO"
            color = "yellow"
            recommendation = "**Ação Sugerida:** Documentar padrão e revisar thresholds"
            priority = "P3"
        else:
            classification = "🟢 NÃO REINCIDENTE"
            level = "BAIXO"
            color = "green"
            recommendation = "**Ação:** Análise caso a caso - possível alarme falso"
            priority = "P4"
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### {classification}")
            st.markdown(f"**Nível:** {level} | **Prioridade:** {priority}")
            st.metric("Score de Reincidência", f"{score}/100", delta=level)
            
            st.markdown("#### 📊 Critérios Avaliados")
            for criterion, points in criteria:
                st.write(f"• {criterion} ({points} pts)")
            
            st.info(recommendation)
            
            st.markdown("#### 📈 Métricas Complementares")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("CV", f"{cv:.2f}")
            col_b.metric("Previsibilidade", f"{pred_score:.0f}%")
            col_c.metric("Anomalias", f"{results.get('anomalies', {}).get('anomaly_rate', 0):.1f}%")
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Score Final", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 50], 'color': "lightyellow"},
                        {'range': [50, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key='final_gauge')
        
        # Predição
        if score >= 50:
            st.markdown("---")
            st.subheader("🔮 Predição de Próxima Ocorrência")
            
            last_alert = df['created_on'].max()
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            pred_time = last_alert + pd.Timedelta(hours=mean_interval)
            conf_interval = 1.96 * std_interval
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Predição", pred_time.strftime('%d/%m %H:%M'))
            col2.metric("Intervalo", f"{mean_interval:.1f}h")
            col3.metric("Confiança (95%)", f"± {conf_interval:.1f}h")
        
        # Exportar
        st.markdown("---")
        export_data = {
            'alert_id': self.alert_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'classificacao': classification,
            'nivel': level,
            'prioridade': priority,
            'score': score,
            'cv': cv,
            'periodicidade': results.get('periodicity', {}).get('has_periodicity', False),
            'autocorrelacao': results.get('autocorr', {}).get('has_autocorr', False),
            'clusters': results.get('clusters', {}).get('n_clusters', 0),
            'previsibilidade': pred_score,
            'anomalias_pct': results.get('anomalies', {}).get('anomaly_rate', 0),
            'recomendacao': recommendation
        }
        
        export_df = pd.DataFrame([export_data])
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="⬇️ Exportar Relatório Completo (CSV)",
            data=csv,
            file_name=f"reincidencia_{self.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# FUNÇÕES AUXILIARES DE AGRUPAMENTO (código original mantido)
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

# ============================================================
# FUNÇÕES DE PROCESSAMENTO (código original mantido)
# ============================================================

def process_single_alert(alert_id, df_original, max_gap_hours=24, min_group_size=3, 
                        spike_threshold_multiplier=5):
    try:
        df_alert = df_original[df_original['u_alert_id'] == alert_id].copy()
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
            'variabilidade_intervalo': intervals_hours.std() / intervals_hours.mean() if len(intervals_hours) > 0 and intervals_hours.mean() > 0 else 0,
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

# ============================================================
# CLASSE PRINCIPAL (mantida com integração da nova análise)
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
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado com {len(df_raw)} registros")
            with st.expander("📋 Informações do Dataset"):
                st.write(f"**Colunas:** {list(df_raw.columns)}")
                st.write(f"**Shape:** {df_raw.shape}")
                st.dataframe(df_raw.head())
            if 'created_on' not in df_raw.columns or 'u_alert_id' not in df_raw.columns:
                st.error("❌ Colunas 'created_on' e 'u_alert_id' são obrigatórias!")
                return False
            df_raw['created_on'] = pd.to_datetime(df_raw['created_on'])
            df_raw = df_raw.dropna(subset=['created_on'])
            df_raw = df_raw.sort_values(['u_alert_id', 'created_on']).reset_index(drop=True)
            self.df_original = df_raw
            st.sidebar.write(f"**IDs disponíveis:** {len(df_raw['u_alert_id'].unique())}")
            return True
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {e}")
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

    def prepare_global_analysis(self, use_multiprocessing=True, max_gap_hours=24, 
                               min_group_size=3, spike_threshold_multiplier=5):
        st.header("🌍 Análise Global de Todos os Alertas")
        self.max_gap_hours = max_gap_hours
        self.min_group_size = min_group_size
        self.spike_threshold_multiplier = spike_threshold_multiplier
        
        unique_ids = self.df_original['u_alert_id'].unique()
        total_ids = len(unique_ids)
        st.info(f"📊 Processando {total_ids} Alert IDs...")
        alert_metrics = []
        
        if use_multiprocessing:
            n_processes = min(cpu_count(), total_ids)
            st.write(f"🚀 Usando {n_processes} processos paralelos")
            chunk_size = max(1, total_ids // n_processes)
            id_chunks = [unique_ids[i:i + chunk_size] for i in range(0, total_ids, chunk_size)]
            progress_bar = st.progress(0)
            status_text = st.empty()
            process_func = partial(process_alert_chunk, 
                                  df_original=self.df_original,
                                  max_gap_hours=max_gap_hours,
                                  min_group_size=min_group_size,
                                  spike_threshold_multiplier=spike_threshold_multiplier)
            try:
                with Pool(processes=n_processes) as pool:
                    results = pool.map(process_func, id_chunks)
                    for result in results:
                        alert_metrics.extend(result)
                    progress_bar.progress(1.0)
                    status_text.success(f"✅ Processamento concluído! {len(alert_metrics)} alertas analisados")
            except Exception as e:
                st.error(f"❌ Erro no multiprocessing: {e}")
                st.warning("⚠️ Tentando processamento sequencial...")
                use_multiprocessing = False
                alert_metrics = []
        
        if not use_multiprocessing or len(alert_metrics) == 0:
            alert_metrics = []
            progress_bar = st.progress(0)
            for i, alert_id in enumerate(unique_ids):
                progress_bar.progress((i + 1) / total_ids)
                metrics = process_single_alert(alert_id, self.df_original, 
                                              max_gap_hours, min_group_size, 
                                              spike_threshold_multiplier)
                if metrics:
                    alert_metrics.append(metrics)
        
        if 'progress_bar' in locals():
            progress_bar.empty()
        
        self.df_all_alerts = pd.DataFrame(alert_metrics)
        
        isolated_count = len(self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'isolated'])
        continuous_count = len(self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'continuous'])
        single_day_count = len(self.df_all_alerts[self.df_all_alerts['unique_days'] == 1])
        
        st.subheader("📊 Estatísticas Globais")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            st.metric("🔢 Total de Alert IDs", len(unique_ids))
        with col2:
            st.metric("📈 IDs com Dados", len(self.df_all_alerts))
        with col3:
            st.metric("🚨 Total de Alertas", self.df_original.shape[0])
        with col4:
            period_total = (self.df_original['created_on'].max() - self.df_original['created_on'].min()).days + 1
            st.metric("📅 Período (dias)", period_total)
        with col5:
            st.metric("🔴 Alertas Isolados", isolated_count)
        with col6:
            st.metric("🟢 Alertas Contínuos", continuous_count)
        with col7:
            st.metric("📆 Alertas de 1 Dia", single_day_count)
        
        return len(self.df_all_alerts) > 0

    # ============================================================
    # MÉTODO ATUALIZADO - ANÁLISE DE RECORRÊNCIA TEMPORAL
    # ============================================================

    def analyze_temporal_recurrence_patterns(self):
        """Análise avançada de recorrência usando a nova classe"""
        analyzer = AdvancedRecurrenceAnalyzer(self.df, self.alert_id)
        analyzer.analyze()

    # ============================================================
    # MÉTODOS RESTANTES (mantidos do código original)
    # ============================================================

    def show_isolated_vs_continuous_analysis(self):
        st.header("🔍 Análise de Alertas Isolados vs Contínuos (Baseado em Grupos)")

        self.df_all_alerts = self.df_all_alerts.drop_duplicates(subset=['alert_id'])

        df_isolated = self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'isolated']
        df_continuous = self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'continuous']
        df_single_day = self.df_all_alerts[self.df_all_alerts['unique_days'] == 1]

        col1, col2 = st.columns(2)
        with col1:
            pattern_dist = self.df_all_alerts['pattern_type'].value_counts()
            fig_pie = px.pie(
                values=pattern_dist.values,
                names=pattern_dist.index,
                title="📊 Distribuição de Padrões de Alerta",
                color_discrete_map={'isolated': '#ff4444', 'continuous': '#44ff44'}
            )
            st.plotly_chart(fig_pie, use_container_width=True, key='pattern_pie')

        with col2:
            st.subheader("📈 Comparação de Métricas")
            comparison_data = pd.DataFrame({
                'Métrica': ['Qtd Alertas', 'Média Ocorrências', 'Média Grupos', 
                            'Média % Isolados', 'Média Freq/Dia', 'Alertas 1 Dia'],
                'Isolados': [
                    len(df_isolated),
                    df_isolated['total_ocorrencias'].mean() if len(df_isolated) > 0 else 0,
                    df_isolated['num_grupos'].mean() if len(df_isolated) > 0 else 0,
                    df_isolated['pct_isolados'].mean() if len(df_isolated) > 0 else 0,
                    df_isolated['freq_dia'].mean() if len(df_isolated) > 0 else 0,
                    len(df_single_day)
                ],
                'Contínuos': [
                    len(df_continuous),
                    df_continuous['total_ocorrencias'].mean() if len(df_continuous) > 0 else 0,
                    df_continuous['num_grupos'].mean() if len(df_continuous) > 0 else 0,
                    df_continuous['pct_isolados'].mean() if len(df_continuous) > 0 else 0,
                    df_continuous['freq_dia'].mean() if len(df_continuous) > 0 else 0,
                    0
                ]
            })
            comparison_data = comparison_data.round(2)
            st.dataframe(comparison_data, use_container_width=True)

        st.subheader("📈 Evolução Temporal: Isolados vs Agrupados")
        
        df_with_dates = self.df_all_alerts.copy()
        df_with_dates['date'] = pd.to_datetime(df_with_dates['primeiro_alerta']).dt.date
        
        daily_isolated = df_isolated.copy()
        daily_isolated['date'] = pd.to_datetime(daily_isolated['primeiro_alerta']).dt.date
        isolated_counts = daily_isolated.groupby('date').size()
        
        daily_continuous = df_continuous.copy()
        daily_continuous['date'] = pd.to_datetime(daily_continuous['primeiro_alerta']).dt.date
        continuous_counts = daily_continuous.groupby('date').size()
        
        all_dates = pd.date_range(
            start=self.df_all_alerts['primeiro_alerta'].min(),
            end=self.df_all_alerts['ultimo_alerta'].max(),
            freq='D'
        ).date
        
        line_data = pd.DataFrame({'date': all_dates})
        line_data['Isolados'] = line_data['date'].map(isolated_counts).fillna(0)
        line_data['Contínuos'] = line_data['date'].map(continuous_counts).fillna(0)
        
        fig_lines = go.Figure()
        
        fig_lines.add_trace(go.Scatter(
            x=line_data['date'],
            y=line_data['Isolados'],
            mode='lines+markers',
            name='Isolados',
            line=dict(color='#ff4444', width=2),
            marker=dict(size=6),
            hovertemplate='%{x}<br>Isolados: %{y}<extra></extra>'
        ))
        
        fig_lines.add_trace(go.Scatter(
            x=line_data['date'],
            y=line_data['Contínuos'],
            mode='lines+markers',
            name='Contínuos',
            line=dict(color='#44ff44', width=2),
            marker=dict(size=6),
            hovertemplate='%{x}<br>Contínuos: %{y}<extra></extra>'
        ))
        
        fig_lines.update_layout(
            title="Quantidade de Alertas por Dia (Isolados vs Contínuos)",
            xaxis_title="Data",
            yaxis_title="Quantidade de Alertas",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_lines, use_container_width=True, key='isolated_vs_continuous_lines')

        tab1, tab2, tab3 = st.tabs(["🔴 Alertas Isolados", "🟢 Alertas Contínuos", "📊 Análise Comparativa"])

        with tab1:
            st.subheader(f"🔴 Alertas Isolados ({len(df_isolated)} alertas)")

            if len(df_isolated) > 0:
                if len(df_single_day) > 0:
                    st.info(f"📆 **{len(df_single_day)} alertas** ({len(df_single_day)/len(df_isolated)*100:.1f}%) ocorreram em apenas 1 dia")
                
                fig_iso = px.scatter(
                    df_isolated,
                    x='primeiro_alerta',
                    y='total_ocorrencias',
                    size='alertas_isolados',
                    color='pct_isolados',
                    title="⏳ Ocorrências de Alertas Isolados no Tempo",
                    hover_data=['alert_id', 'pattern_reason', 'num_grupos', 'unique_days'],
                    labels={'pct_isolados': '% Isolados', 'unique_days': 'Dias Únicos'}
                )
                st.plotly_chart(fig_iso, use_container_width=True, key='isolated_scatter')

                st.write("**📝 Razões para Classificação como Isolado:**")
                reason_counts = df_isolated['pattern_reason'].value_counts()
                for reason, count in reason_counts.items():
                    st.write(f"• {reason}: {count} alertas")

                st.write("**🔝 Top 10 Alertas Isolados (por % de alertas isolados):**")
                top_isolated = df_isolated.nlargest(10, 'pct_isolados')[
                    ['alert_id', 'total_ocorrencias', 'alertas_isolados', 'num_grupos', 'pct_isolados', 'unique_days', 'pattern_reason']
                ]
                top_isolated.columns = ['Alert ID', 'Total Ocorrências', 'Alertas Isolados', 'Nº Grupos', '% Isolados', 'Dias Únicos', 'Razão']
                top_isolated['% Isolados'] = top_isolated['% Isolados'].round(1).astype(str) + '%'
                st.dataframe(top_isolated, use_container_width=True)

                with st.expander("📋 Ver todos os alertas isolados"):
                    isolated_list = df_isolated[['alert_id', 'total_ocorrencias', 'alertas_isolados',
                                                'num_grupos', 'pct_isolados', 'unique_days', 'pattern_reason']].copy()
                    isolated_list.columns = ['Alert ID', 'Total', 'Isolados', 'Grupos', '% Isolados', 'Dias Únicos', 'Razão']
                    isolated_list['% Isolados'] = isolated_list['% Isolados'].round(1).astype(str) + '%'
                    st.dataframe(isolated_list, use_container_width=True)
            else:
                st.info("Nenhum alerta isolado encontrado com os critérios atuais.")

        with tab2:
            st.subheader(f"🟢 Alertas Contínuos ({len(df_continuous)} alertas)")

            if len(df_continuous) > 0:
                st.write("**🔝 Top 10 Alertas Contínuos (maior número de grupos):**")
                top_continuous = df_continuous.nlargest(10, 'num_grupos')[
                    ['alert_id', 'total_ocorrencias', 'num_grupos', 'alertas_agrupados', 'freq_dia', 'unique_days']
                ]
                top_continuous.columns = ['Alert ID', 'Total Ocorrências', 'Nº Grupos', 'Alertas Agrupados', 'Freq/Dia', 'Dias Únicos']
                st.dataframe(top_continuous, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig_groups = px.histogram(
                        df_continuous, 
                        x='num_grupos',
                        title="📊 Distribuição de Número de Grupos",
                        labels={'num_grupos': 'Número de Grupos', 'count': 'Quantidade'}
                    )
                    st.plotly_chart(fig_groups, use_container_width=True, key='continuous_groups_hist')
                with col2:
                    fig_pct = px.histogram(
                        df_continuous,
                        x='pct_isolados',
                        title="📊 Distribuição de % de Alertas Isolados",
                        labels={'pct_isolados': '% Alertas Isolados', 'count': 'Quantidade'}
                    )
                    st.plotly_chart(fig_pct, use_container_width=True, key='continuous_pct_hist')

                with st.expander("📋 Ver todos os alertas contínuos"):
                    continuous_list = df_continuous[['alert_id', 'total_ocorrencias', 'num_grupos',
                                                    'alertas_agrupados', 'alertas_isolados', 'pct_isolados', 'unique_days']].copy()
                    continuous_list.columns = ['Alert ID', 'Total', 'Grupos', 'Agrupados', 'Isolados', '% Isolados', 'Dias Únicos']
                    continuous_list['% Isolados'] = continuous_list['% Isolados'].round(1).astype(str) + '%'
                    st.dataframe(continuous_list, use_container_width=True)
            else:
                st.info("Nenhum alerta contínuo encontrado com os critérios atuais.")

        with tab3:
            st.subheader("📊 Análise Comparativa Detalhada")

            fig_scatter = px.scatter(
                self.df_all_alerts,
                x='total_ocorrencias',
                y='intervalo_medio_h',
                color='pattern_type',
                title="🎯 Ocorrências vs Intervalo Médio",
                labels={
                    'total_ocorrencias': 'Total de Ocorrências',
                    'intervalo_medio_h': 'Intervalo Médio (horas)',
                    'pattern_type': 'Tipo de Padrão'
                },
                hover_data=['alert_id', 'unique_days'],
                color_discrete_map={'isolated': '#ff4444', 'continuous': '#44ff44'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key='comparative_scatter')

            col1, col2 = st.columns(2)
            with col1:
                fig_box_occ = px.box(
                    self.df_all_alerts,
                    x='pattern_type',
                    y='total_ocorrencias',
                    title="📦 Distribuição de Ocorrências",
                    color='pattern_type',
                    color_discrete_map={'isolated': '#ff4444', 'continuous': '#44ff44'}
                )
                st.plotly_chart(fig_box_occ, use_container_width=True, key='box_occurrences')

            with col2:
                fig_box_freq = px.box(
                    self.df_all_alerts,
                    x='pattern_type',
                    y='freq_dia',
                    title="📦 Distribuição de Frequência Diária",
                    color='pattern_type',
                    color_discrete_map={'isolated': '#ff4444', 'continuous': '#44ff44'}
                )
                st.plotly_chart(fig_box_freq, use_container_width=True, key='box_frequency')

            st.subheader("💡 Recomendações de Tratamento")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**🔴 Para Alertas Isolados:**")
                st.write("• Considerar desativação ou revisão de configuração")
                st.write("• Verificar se são falsos positivos")
                st.write("• Analisar contexto específico das ocorrências")
                st.write("• Avaliar consolidação com outros alertas similares")
                st.write("• Alertas de 1 dia podem ser eventos únicos sem recorrência")

            with col2:
                st.write("**🟢 Para Alertas Contínuos:**")
                st.write("• Priorizar automação de resposta")
                st.write("• Implementar supressão inteligente")
                st.write("• Criar runbooks específicos")
                st.write("• Considerar ajuste de thresholds")

    def show_continuous_groups_detailed_view(self):
        st.header("🔍 Visualização Detalhada dos Grupos - Alertas Contínuos")
        
        df_continuous = self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'continuous']
        
        if len(df_continuous) == 0:
            st.warning("⚠️ Nenhum alerta contínuo encontrado para visualização de grupos.")
            return
        
        st.info(f"📊 Analisando grupos detalhados de **{len(df_continuous)}** alertas contínuos")
        
        selected_alerts = st.multiselect(
            "🎯 Selecione alertas para visualizar grupos em detalhes (máx. 5):",
            options=df_continuous.nlargest(20, 'num_grupos')['alert_id'].tolist(),
            default=df_continuous.nlargest(3, 'num_grupos')['alert_id'].tolist()[:3],
            help="Mostrando os 20 alertas com mais grupos. Selecione até 5 para análise detalhada."
        )
        
        if len(selected_alerts) > 5:
            st.warning("⚠️ Máximo de 5 alertas por vez. Mostrando apenas os 5 primeiros selecionados.")
            selected_alerts = selected_alerts[:5]
        
        if not selected_alerts:
            st.info("👆 Selecione pelo menos um alerta acima para ver os detalhes dos grupos")
            return
        
        for alert_id in selected_alerts:
            st.markdown("---")
            alert_info = df_continuous[df_continuous['alert_id'] == alert_id].iloc[0]
            
            with st.expander(f"📊 **Alert ID: {alert_id}** - {alert_info['num_grupos']} grupos identificados", expanded=True):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Ocorrências", alert_info['total_ocorrencias'])
                with col2:
                    st.metric("Nº de Grupos", alert_info['num_grupos'])
                with col3:
                    st.metric("Alertas Agrupados", alert_info['alertas_agrupados'])
                with col4:
                    st.metric("Alertas Isolados", alert_info['alertas_isolados'])
                with col5:
                    st.metric("Dias Únicos", alert_info['unique_days'])
                
                alert_data = self.df_original[self.df_original['u_alert_id'] == alert_id].copy()
                alert_data, groups_info = identify_alert_groups(
                    alert_data,
                    self.max_gap_hours,
                    self.min_group_size,
                    self.spike_threshold_multiplier
                )
                
                if len(groups_info) > 0:
                    st.subheader("📋 Detalhes dos Grupos Identificados")
                    groups_df = pd.DataFrame(groups_info)
                    groups_df['start_time_str'] = pd.to_datetime(groups_df['start_time']).dt.strftime('%Y-%m-%d %H:%M')
                    groups_df['end_time_str'] = pd.to_datetime(groups_df['end_time']).dt.strftime('%Y-%m-%d %H:%M')
                    groups_df['duration_hours'] = groups_df['duration_hours'].round(2)
                    
                    groups_display = groups_df[['group_id', 'size', 'start_time_str', 'end_time_str', 'duration_hours']].copy()
                    groups_display.columns = ['ID Grupo', 'Tamanho', 'Início', 'Fim', 'Duração (h)']
                    st.dataframe(groups_display, use_container_width=True)
                    
                    st.subheader("📊 Timeline Visual dos Grupos")
                    
                    fig_timeline = go.Figure()
                    
                    colors = px.colors.qualitative.Plotly
                    for idx, group in groups_df.iterrows():
                        color = colors[int(group['group_id']) % len(colors)]
                        
                        fig_timeline.add_trace(go.Scatter(
                            x=[group['start_time'], group['end_time']],
                            y=[group['group_id'], group['group_id']],
                            mode='lines+markers',
                            name=f"Grupo {int(group['group_id'])}",
                            line=dict(color=color, width=15),
                            marker=dict(size=12, symbol='circle'),
                            hovertemplate=f"<b>Grupo {int(group['group_id'])}</b><br>" +
                                        f"Tamanho: {group['size']} alertas<br>" +
                                        f"Duração: {group['duration_hours']:.2f}h<br>" +
                                        f"Início: {group['start_time_str']}<br>" +
                                        f"Fim: {group['end_time_str']}<extra></extra>"
                        ))
                    
                    isolated_data = alert_data[alert_data['is_isolated']]
                    if len(isolated_data) > 0:
                        fig_timeline.add_trace(go.Scatter(
                            x=isolated_data['created_on'],
                            y=[-1] * len(isolated_data),
                            mode='markers',
                            name='Alertas Isolados',
                            marker=dict(size=10, color='red', symbol='x'),
                            hovertemplate='<b>Alerta Isolado</b><br>%{x}<extra></extra>'
                        ))
                    
                    fig_timeline.update_layout(
                        title=f"Timeline de Grupos - Alert ID: {alert_id}",
                        xaxis_title="Data/Hora",
                        yaxis_title="ID do Grupo",
                        yaxis=dict(
                            tickmode='linear',
                            tick0=-1,
                            dtick=1,
                            ticktext=['Isolados'] + [f'Grupo {i}' for i in range(len(groups_info))],
                            tickvals=[-1] + list(range(len(groups_info)))
                        ),
                        height=400,
                        hovermode='closest',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_timeline, use_container_width=True, key=f'timeline_{alert_id}')
                    
                    st.subheader("📈 Análise Temporal dos Grupos")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_sizes = px.bar(
                            groups_df,
                            x='group_id',
                            y='size',
                            title="Tamanho de Cada Grupo",
                            labels={'group_id': 'ID do Grupo', 'size': 'Quantidade de Alertas'},
                            text='size'
                        )
                        fig_sizes.update_traces(textposition='outside')
                        st.plotly_chart(fig_sizes, use_container_width=True, key=f'sizes_{alert_id}')
                    
                    with col2:
                        fig_duration = px.bar(
                            groups_df,
                            x='group_id',
                            y='duration_hours',
                            title="Duração de Cada Grupo (horas)",
                            labels={'group_id': 'ID do Grupo', 'duration_hours': 'Duração (h)'},
                            text='duration_hours'
                        )
                        fig_duration.update_traces(textposition='outside', texttemplate='%{text:.1f}h')
                        st.plotly_chart(fig_duration, use_container_width=True, key=f'duration_{alert_id}')
                    
                    st.subheader("📊 Estatísticas dos Grupos")
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    
                    with stats_col1:
                        st.metric("Tamanho Médio", f"{groups_df['size'].mean():.1f} alertas")
                    with stats_col2:
                        st.metric("Maior Grupo", f"{groups_df['size'].max()} alertas")
                    with stats_col3:
                        st.metric("Duração Média", f"{groups_df['duration_hours'].mean():.1f}h")
                    with stats_col4:
                        st.metric("Maior Duração", f"{groups_df['duration_hours'].max():.1f}h")
                    
                    if len(groups_df) > 1:
                        st.subheader("⏱️ Intervalos Entre Grupos")
                        gaps = []
                        for i in range(len(groups_df) - 1):
                            gap = (groups_df.iloc[i+1]['start_time'] - groups_df.iloc[i]['end_time']).total_seconds() / 3600
                            gaps.append({
                                'De': f"Grupo {int(groups_df.iloc[i]['group_id'])}",
                                'Para': f"Grupo {int(groups_df.iloc[i+1]['group_id'])}",
                                'Intervalo (h)': round(gap, 2)
                            })
                        
                        gaps_df = pd.DataFrame(gaps)
                        st.dataframe(gaps_df, use_container_width=True)
                        
                        avg_gap = gaps_df['Intervalo (h)'].mean()
                        st.info(f"📊 Intervalo médio entre grupos: **{avg_gap:.2f} horas**")
                
                else:
                    st.warning("Nenhum grupo identificado para este alerta.")
        
        st.markdown("---")
        st.header("📊 Resumo Geral dos Grupos - Todos os Alertas Contínuos")
        
        all_groups_data = []
        for _, alert in df_continuous.iterrows():
            alert_data = self.df_original[self.df_original['u_alert_id'] == alert['alert_id']].copy()
            _, groups_info = identify_alert_groups(
                alert_data,
                self.max_gap_hours,
                self.min_group_size,
                self.spike_threshold_multiplier
            )
            for group in groups_info:
                all_groups_data.append({
                    'alert_id': alert['alert_id'],
                    'group_id': group['group_id'],
                    'size': group['size'],
                    'duration_hours': group['duration_hours']
                })
        
        if all_groups_data:
            all_groups_df = pd.DataFrame(all_groups_data)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Grupos", len(all_groups_df))
            with col2:
                st.metric("Tamanho Médio", f"{all_groups_df['size'].mean():.1f} alertas")
            with col3:
                st.metric("Duração Média", f"{all_groups_df['duration_hours'].mean():.1f}h")
            with col4:
                st.metric("Alertas/Grupo Máx", int(all_groups_df['size'].max()))
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_all_sizes = px.histogram(
                    all_groups_df,
                    x='size',
                    title="Distribuição de Tamanhos dos Grupos",
                    labels={'size': 'Tamanho do Grupo (alertas)', 'count': 'Quantidade de Grupos'},
                    nbins=20
                )
                st.plotly_chart(fig_all_sizes, use_container_width=True, key='all_sizes_hist')
            
            with col2:
                fig_all_duration = px.histogram(
                    all_groups_df,
                    x='duration_hours',
                    title="Distribuição de Durações dos Grupos",
                    labels={'duration_hours': 'Duração (horas)', 'count': 'Quantidade de Grupos'},
                    nbins=20
                )
                st.plotly_chart(fig_all_duration, use_container_width=True, key='all_duration_hist')

    def analyze_continuous_recurrence_patterns(self):
        st.header("🔁 Análise de Recorrência - Alertas Contínuos")

        df_continuous = self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'continuous']

        if len(df_continuous) == 0:
            st.warning("⚠️ Nenhum alerta contínuo encontrado para análise de recorrência.")
            return

        st.info(f"📊 Analisando padrões de recorrência de **{len(df_continuous)}** alertas contínuos")

        continuous_alert_ids = df_continuous['alert_id'].unique()
        df_continuous_details = self.df_original[self.df_original['u_alert_id'].isin(continuous_alert_ids)].copy()

        df_continuous_details['hour'] = df_continuous_details['created_on'].dt.hour
        df_continuous_details['day_of_week'] = df_continuous_details['created_on'].dt.dayofweek
        df_continuous_details['day_name'] = df_continuous_details['created_on'].dt.day_name()

        # ============================================================
        # ⏰ PADRÃO DE RECORRÊNCIA POR HORA
        # ============================================================
        st.subheader("⏰ Padrão de Recorrência por Hora do Dia")

        hourly_dist = df_continuous_details['hour'].value_counts().sort_index()
        hourly_pct = (hourly_dist / hourly_dist.sum() * 100).round(2)

        top_3_hours = hourly_pct.nlargest(3)
        total_top_3_hours = top_3_hours.sum()

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Bar(
                x=hourly_dist.index,
                y=hourly_dist.values,
                marker_color=['red' if i in top_3_hours.index else 'lightblue'
                             for i in hourly_dist.index],
                text=hourly_pct.values,
                texttemplate='%{text:.1f}%',
                textposition='outside',
                hovertemplate='Hora: %{x}:00<br>Alertas: %{y}<br>% do total: %{text:.1f}%<extra></extra>'
            ))
            fig_hourly.update_layout(
                title="Distribuição de Alertas Contínuos por Hora",
                xaxis_title="Hora do Dia",
                yaxis_title="Quantidade de Alertas",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_hourly, use_container_width=True, key='recurrence_hourly')

        with col2:
            st.metric("🕐 Hora com Mais Alertas", f"{top_3_hours.index[0]}:00")
            st.metric("📊 % nesta Hora", f"{top_3_hours.values[0]:.1f}%")
            st.metric("🔝 Top 3 Horas (% total)", f"{total_top_3_hours:.1f}%")

            # ✅ Classificação de concentração horária
            if total_top_3_hours > 60:
                pattern_hour = "🔴 **Concentrado**"
                hour_desc = "Alertas altamente concentrados em poucas horas"
            elif total_top_3_hours > 40:
                pattern_hour = "🟡 **Moderado**"
                hour_desc = "Alertas parcialmente concentrados"
            else:
                pattern_hour = "🟢 **Distribuído**"
                hour_desc = "Alertas bem distribuídos ao longo do dia"

            st.write(f"**Padrão:** {pattern_hour}")
            st.write(hour_desc)

        st.write("**🔝 Top 5 Horários:**")
        top_5_hours = hourly_pct.nlargest(5)
        for hour, pct in top_5_hours.items():
            st.write(f"• **{hour:02d}:00** - {hourly_dist[hour]} alertas ({pct:.1f}%)")

        st.markdown("---")

        # ============================================================
        # 📅 PADRÃO DE RECORRÊNCIA POR DIA DA SEMANA
        # ============================================================
        st.subheader("📅 Padrão de Recorrência por Dia da Semana")

        daily_dist = df_continuous_details['day_name'].value_counts()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_dist_ordered = daily_dist.reindex(days_order).fillna(0)
        daily_pct = (daily_dist_ordered / daily_dist_ordered.sum() * 100).round(2)

        top_3_days = daily_pct.nlargest(3)
        total_top_3_days = top_3_days.sum()

        day_translation = {
            'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
            'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        }

        daily_pct_pt = daily_pct.rename(index=day_translation)
        daily_dist_ordered_pt = daily_dist_ordered.rename(index=day_translation)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_daily = go.Figure()
            fig_daily.add_trace(go.Bar(
                x=list(daily_pct_pt.index),
                y=daily_dist_ordered_pt.values,
                marker_color=['red' if day in [day_translation[d] for d in top_3_days.index] else 'lightgreen'
                             for day in daily_pct_pt.index],
                text=daily_pct_pt.values,
                texttemplate='%{text:.1f}%',
                textposition='outside',
                hovertemplate='Dia: %{x}<br>Alertas: %{y}<br>% do total: %{text:.1f}%<extra></extra>'
            ))
            fig_daily.update_layout(
                title="Distribuição de Alertas Contínuos por Dia da Semana",
                xaxis_title="Dia da Semana",
                yaxis_title="Quantidade de Alertas",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_daily, use_container_width=True, key='recurrence_daily')

        with col2:
            top_day_en = top_3_days.index[0]
            top_day_pt = day_translation[top_day_en]
            st.metric("📅 Dia com Mais Alertas", top_day_pt)
            st.metric("📊 % neste Dia", f"{top_3_days.values[0]:.1f}%")
            st.metric("🔝 Top 3 Dias (% total)", f"{total_top_3_days:.1f}%")

            if total_top_3_days > 60:
                pattern_day = "🔴 **Concentrado**"
                day_desc = "Alertas altamente concentrados em poucos dias"
            elif total_top_3_days > 45:
                pattern_day = "🟡 **Moderado**"
                day_desc = "Alertas parcialmente concentrados"
            else:
                pattern_day = "🟢 **Distribuído**"
                day_desc = "Alertas bem distribuídos na semana"

            st.write(f"**Padrão:** {pattern_day}")
            st.write(day_desc)

        st.write("**🔝 Ranking de Dias:**")
        top_days_sorted = daily_pct.sort_values(ascending=False)
        for day, pct in top_days_sorted.items():
            day_pt = day_translation[day]
            count = daily_dist_ordered[day]
            st.write(f"• **{day_pt}** - {int(count)} alertas ({pct:.1f}%)")

        st.markdown("---")

        # ============================================================
        # 🎯 RESUMO DO PADRÃO DE RECORRÊNCIA
        # ============================================================
        st.subheader("🎯 Resumo do Padrão de Recorrência")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**⏰ Padrão Horário:**")
            st.write(f"• {pattern_hour}")
            st.write(f"• Top 3 horas concentram {total_top_3_hours:.1f}% dos alertas")
            st.write(f"• Horário principal: **{top_3_hours.index[0]:02d}:00**")

            if total_top_3_hours > 50:
                st.write("💡 **Recomendação:** Avaliar janela de manutenção neste horário")

        with col2:
            st.write("**📅 Padrão Semanal:**")
            st.write(f"• {pattern_day}")
            st.write(f"• Top 3 dias concentram {total_top_3_days:.1f}% dos alertas")
            st.write(f"• Dia principal: **{day_translation[top_day_en]}**")

            if total_top_3_days > 50:
                st.write("💡 **Recomendação:** Atenção redobrada nestes dias")

        st.markdown("---")
        st.subheader("🏆 Padrão Dominante")


        if total_top_3_hours > total_top_3_days:
            st.success(f"⏰ **HORA DO DIA** é o padrão dominante ({total_top_3_hours:.1f}% vs {total_top_3_days:.1f}%)")
            st.write(f"Os alertas contínuos tendem a ocorrer principalmente no horário das **{top_3_hours.index[0]:02d}:00**")
        elif total_top_3_days > total_top_3_hours:
            st.success(f"📅 **DIA DA SEMANA** é o padrão dominante ({total_top_3_days:.1f}% vs {total_top_3_hours:.1f}%)")
            st.write(f"Os alertas contínuos tendem a ocorrer principalmente às **{day_translation[top_day_en]}**")
        else:
            st.info("📊 **Padrão BALANCEADO** - Não há concentração clara em hora ou dia específicos")

        st.markdown("---")

        # ============================================================
        # 🔥 MAPA DE CALOR - HORA × DIA DA SEMANA
        # ============================================================
        st.subheader("🔥 Mapa de Calor: Hora × Dia da Semana")

        heatmap_data = df_continuous_details.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        heatmap_pivot = heatmap_data.pivot(index='hour', columns='day_of_week', values='count').fillna(0)

        day_map = {0: 'Seg', 1: 'Ter', 2: 'Qua', 3: 'Qui', 4: 'Sex', 5: 'Sáb', 6: 'Dom'}
        heatmap_pivot.columns = [day_map[col] for col in heatmap_pivot.columns]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='Reds',
            hovertemplate='Dia: %{x}<br>Hora: %{y}:00<br>Alertas: %{z}<extra></extra>'
        ))

        fig_heatmap.update_layout(
            title="Concentração de Alertas por Dia e Hora",
            xaxis_title="Dia da Semana",
            yaxis_title="Hora do Dia",
            height=600
        )

        st.plotly_chart(fig_heatmap, use_container_width=True, key='recurrence_heatmap')

        # ============================================================
        # ANÁLISE GLOBAL - VISÃO GERAL
        # ============================================================

    def show_global_overview(self):
        st.subheader("📈 Visão Geral dos Alertas")
        
        df_to_analyze = self.df_all_alerts
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🔥 Top 10 Alertas Mais Frequentes**")
            top_frequent = df_to_analyze.nlargest(10, 'total_ocorrencias')[['alert_id', 'total_ocorrencias', 'freq_dia', 'pattern_type', 'unique_days']]
            top_frequent.columns = ['Alert ID', 'Total Ocorrências', 'Frequência/Dia', 'Tipo', 'Dias Únicos']
            st.dataframe(top_frequent, use_container_width=True)
        with col2:
            st.write("**⚡ Top 10 Alertas Mais Rápidos (Menor Intervalo)**")
            df_with_intervals = df_to_analyze.dropna(subset=['intervalo_medio_h'])
            if len(df_with_intervals) > 0:
                top_fast = df_with_intervals.nsmallest(10, 'intervalo_medio_h')[['alert_id', 'intervalo_medio_h', 'total_ocorrencias', 'pattern_type']]
                top_fast.columns = ['Alert ID', 'Intervalo Médio (h)', 'Total Ocorrências', 'Tipo']
                st.dataframe(top_fast, use_container_width=True)
            else:
                st.info("Sem dados de intervalo disponíveis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fig_freq = px.histogram(df_to_analyze, x='freq_dia', title="📊 Distribuição de Frequência (alertas/dia)",
                                   labels={'freq_dia': 'Alertas por Dia', 'count': 'Quantidade de Alert IDs'})
            st.plotly_chart(fig_freq, use_container_width=True)
        with col2:
            fig_int = px.histogram(df_to_analyze, x='freq_semana', title="📊 Distribuição de Frequência (alertas/semana)",
                                  labels={'freq_semana': 'Alertas por semana', 'count': 'Quantidade de Alert IDs'})
            st.plotly_chart(fig_int, use_container_width=True)
        with col3:
            fig_int = px.histogram(df_to_analyze, x='freq_mes', title="📊 Distribuição de Frequência (alertas/mês)",
                                  labels={'freq_mes': 'Alertas por mês', 'count': 'Quantidade de Alert IDs'})
            st.plotly_chart(fig_int, use_container_width=True)
        with col4:
            df_with_intervals = df_to_analyze.dropna(subset=['intervalo_medio_h'])
            if len(df_with_intervals) > 0:
                fig_int = px.histogram(df_with_intervals, x='intervalo_medio_h', title="⏱️ Distribuição de Intervalos Médios",
                                      labels={'intervalo_medio_h': 'Intervalo Médio (horas)', 'count': 'Quantidade de Alert IDs'})
                st.plotly_chart(fig_int, use_container_width=True)

    def perform_clustering_analysis(self, use_only_continuous=True):
        st.subheader("🎯 Agrupamento de Alertas por Perfil de Comportamento")
        
        df_for_clustering = self.df_all_alerts
        if use_only_continuous:
            df_for_clustering = self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'continuous'].copy()
            st.info(f"🔍 Usando apenas alertas contínuos para clustering ({len(df_for_clustering)} alertas)")
        
        if len(df_for_clustering) < 2:
            st.warning("⚠️ Dados insuficientes para clustering")
            return None
        
        features = [
            'freq_dia', 'intervalo_medio_h', 'intervalo_std_h',
            'hora_pico', 'pct_fins_semana', 'pct_horario_comercial', 'variabilidade_intervalo'
        ]
        X = df_for_clustering[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        st.write("**🔍 Determinando Número Ótimo de Clusters...**")
        max_clusters = min(10, len(X) - 1)
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        
        optimal_k = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Número Ótimo de Clusters", optimal_k)
        with col2:
            st.metric("📊 Silhouette Score", f"{max(silhouette_scores):.3f}")
        
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans_final.fit_predict(X_scaled)
        
        df_for_clustering['cluster'] = clusters
        
        self.df_all_alerts['cluster'] = np.nan
        self.df_all_alerts.loc[df_for_clustering.index, 'cluster'] = df_for_clustering['cluster']
        
        col1, col2 = st.columns(2)
        with col1:
            fig_scatter = px.scatter(
                df_for_clustering,
                x='freq_dia',
                y='intervalo_medio_h',
                color='cluster',
                size='total_ocorrencias',
                hover_data=['alert_id'],
                title="🎨 Clusters: Frequência vs Intervalo Médio"
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key='cluster_scatter')
        with col2:
            cluster_dist = df_for_clustering['cluster'].value_counts().sort_index()
            fig_dist = px.bar(
                x=cluster_dist.index,
                y=cluster_dist.values,
                title="📊 Distribuição de Alertas por Cluster",
                labels={'x': 'Cluster', 'y': 'Quantidade de Alert IDs'}
            )
            st.plotly_chart(fig_dist, use_container_width=True, key='cluster_dist')
        return optimal_k

    def show_cluster_profiles(self, n_clusters):
        st.subheader("👥 Perfis dos Clusters")
        cluster_tabs = st.tabs([f"Cluster {i}" for i in range(n_clusters)])
        for i in range(n_clusters):
            with cluster_tabs[i]:
                cluster_data = self.df_all_alerts[self.df_all_alerts['cluster'] == i]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Quantidade de Alertas", len(cluster_data))
                with col2:
                    avg_freq = cluster_data['freq_dia'].mean()
                    st.metric("📈 Freq. Média/Dia", f"{avg_freq:.2f}")
                with col3:
                    avg_interval = cluster_data['intervalo_medio_h'].mean()
                    st.metric("⏱️ Intervalo Médio (h)", f"{avg_interval:.2f}")
                with col4:
                    avg_hour = cluster_data['hora_pico'].mean()
                    st.metric("🕐 Hora Pico Média", f"{avg_hour:.0f}:00")
                st.write("**🎯 Características do Cluster:**")
                weekend_pct = cluster_data['pct_fins_semana'].mean()
                business_pct = cluster_data['pct_horario_comercial'].mean()
                characteristics = []
                if avg_freq > self.df_all_alerts['freq_dia'].median():
                    characteristics.append("🔥 **Alta frequência**")
                else:
                    characteristics.append("🐌 **Baixa frequência**")
                if avg_interval < self.df_all_alerts['intervalo_medio_h'].median():
                    characteristics.append("⚡ **Intervalos curtos**")
                else:
                    characteristics.append("⏳ **Intervalos longos**")
                if weekend_pct > 30:
                    characteristics.append("🗓️ **Ativo nos fins de semana**")
                if business_pct > 70:
                    characteristics.append("🏢 **Predominantemente em horário comercial**")
                elif business_pct < 30:
                    characteristics.append("🌙 **Predominantemente fora do horário comercial**")
                for char in characteristics:
                    st.write(f"• {char}")
                with st.expander(f"📋 Alertas no Cluster {i}"):
                    cluster_alerts = cluster_data[['alert_id', 'total_ocorrencias', 'freq_dia', 'intervalo_medio_h']].copy()
                    cluster_alerts.columns = ['Alert ID', 'Total Ocorrências', 'Freq/Dia', 'Intervalo Médio (h)']
                    st.dataframe(cluster_alerts, use_container_width=True, key=f'cluster_table_{i}')

    def show_cluster_recommendations(self):
        st.subheader("💡 Recomendações por Cluster")
        for cluster_id in sorted(self.df_all_alerts['cluster'].dropna().unique()):
            cluster_data = self.df_all_alerts[self.df_all_alerts['cluster'] == cluster_id]
            avg_freq = cluster_data['freq_dia'].mean()
            avg_interval = cluster_data['intervalo_medio_h'].mean()
            weekend_pct = cluster_data['pct_fins_semana'].mean()
            business_pct = cluster_data['pct_horario_comercial'].mean()
            with st.expander(f"🎯 Recomendações para Cluster {int(cluster_id)} ({len(cluster_data)} alertas)"):
                recommendations = []
                if avg_freq > 5:
                    recommendations.append("🚨 **Prioridade Alta**: Alertas muito frequentes - investigar causa raiz")
                    recommendations.append("🔧 **Ação**: Considerar automação de resposta ou ajuste de thresholds")
                if avg_interval < 1:
                    recommendations.append("⚡ **Rajadas detectadas**: Possível tempestade de alertas")
                    recommendations.append("🛡️ **Ação**: Implementar rate limiting ou supressão inteligente")
                if weekend_pct > 50:
                    recommendations.append("🗓️ **Padrão de fim de semana**: Alertas ativos nos fins de semana")
                    recommendations.append("👥 **Ação**: Verificar cobertura de plantão")
                if business_pct < 30:
                    recommendations.append("🌙 **Padrão noturno**: Principalmente fora do horário comercial")
                    recommendations.append("🔄 **Ação**: Considerar processos automatizados noturnos")
                if avg_freq < 0.5:
                    recommendations.append("📉 **Baixa frequência**: Alertas esporádicos")
                    recommendations.append("📊 **Ação**: Revisar relevância e configuração do alerta")
                for rec in recommendations:
                    st.write(f"• {rec}")
                if not recommendations:
                    st.write("• ✅ **Padrão normal**: Nenhuma ação específica recomendada")

    def show_basic_stats(self):
        st.header("📊 Estatísticas Básicas")
        total = len(self.df)
        period_days = (self.dates.max() - self.dates.min()).days + 1
        avg_per_day = total / period_days
        unique_days = self.df['date'].nunique()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🔥 Total de Ocorrências", total)
        with col2:
            st.metric("📅 Período (dias)", period_days)
        with col3:
            st.metric("📆 Dias Únicos", unique_days)
        with col4:
            st.metric("📈 Média/dia", f"{avg_per_day:.2f}")
        with col5:
            last_alert = self.dates.max().strftime("%d/%m %H:%M")
            st.metric("🕐 Último Alerta", last_alert)
        
        if unique_days == 1:
            st.warning("⚠️ **ATENÇÃO:** Todos os alertas ocorreram em apenas 1 dia! Este alerta é classificado como ISOLADO.")
        
        intervals = self.df['time_diff_hours'].dropna()
        if len(intervals) > 0:
            st.subheader("⏱️ Intervalos Entre Alertas")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Média (h)", f"{intervals.mean():.2f}")
            with col2:
                st.metric("Mediana (h)", f"{intervals.median():.2f}")
            with col3:
                st.metric("Mínimo (h)", f"{intervals.min():.2f}")
            with col4:
                st.metric("Máximo (h)", f"{intervals.max():.2f}")

    def show_individual_alert_analysis(self):
        st.header(f"📌 Análise Individual do Alert ID: {self.alert_id}")

        if self.df is None or len(self.df) == 0:
            st.info("Nenhum dado disponível para este alerta.")
            return

        unique_days = self.df['date'].nunique()
        is_single_day = unique_days == 1

        df_isolated = self.df[self.df['is_isolated']]
        df_grouped = self.df[~self.df['is_isolated']]

        st.subheader("📊 Estatísticas Gerais do Alert ID")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Total Ocorrências", len(self.df))
        with col2:
            st.metric("🔴 Isolados", len(df_isolated))
        with col3:
            st.metric("🟢 Agrupados", len(df_grouped))
        with col4:
            st.metric("📦 Nº de Grupos", len(self.groups_info))
        with col5:
            pct_isolated = (len(df_isolated) / len(self.df) * 100) if len(self.df) > 0 else 0
            st.metric("% Isolados", f"{pct_isolated:.1f}%")
        with col6:
            st.metric("📆 Dias Únicos", unique_days)

        if is_single_day:
            st.warning("⚠️ **ATENÇÃO:** Todos os alertas ocorreram em apenas 1 dia! Este padrão é classificado como ISOLADO.")
            st.info(f"📅 Data única: {self.df['date'].iloc[0]}")

        if len(self.groups_info) > 0:
            st.subheader("📦 Informações dos Grupos")
            groups_df = pd.DataFrame(self.groups_info)
            groups_df['start_time'] = pd.to_datetime(groups_df['start_time']).dt.strftime('%Y-%m-%d %H:%M')
            groups_df['end_time'] = pd.to_datetime(groups_df['end_time']).dt.strftime('%Y-%m-%d %H:%M')
            groups_df['duration_hours'] = groups_df['duration_hours'].round(2)
            groups_df.columns = ['ID Grupo', 'Tamanho', 'Início', 'Fim', 'Duração (h)']
            st.dataframe(groups_df, use_container_width=True)

        st.subheader("📈 Gráfico de Linhas: Alertas ao Longo do Tempo")

        df_daily = self.df.groupby(['date', 'is_isolated']).size().reset_index(name='count')
        df_daily_pivot = df_daily.pivot(index='date', columns='is_isolated', values='count').fillna(0)

        new_column_names = {}
        if False in df_daily_pivot.columns:
            new_column_names[False] = 'Agrupados'
        if True in df_daily_pivot.columns:
            new_column_names[True] = 'Isolados'

        df_daily_pivot = df_daily_pivot.rename(columns=new_column_names)

        if 'Agrupados' not in df_daily_pivot.columns:
            df_daily_pivot['Agrupados'] = 0
        if 'Isolados' not in df_daily_pivot.columns:
            df_daily_pivot['Isolados'] = 0

        fig_timeline = go.Figure()

        fig_timeline.add_trace(go.Scatter(
            x=df_daily_pivot.index,
            y=df_daily_pivot['Isolados'],
            mode='lines+markers',
            name='Isolados',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.3)'
        ))

        fig_timeline.add_trace(go.Scatter(
            x=df_daily_pivot.index,
            y=df_daily_pivot['Agrupados'],
            mode='lines+markers',
            name='Agrupados',
            line=dict(color='green', width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(68, 255, 68, 0.3)'
        ))

        fig_timeline.update_layout(
            title="Evolução Diária: Alertas Isolados vs Agrupados",
            xaxis_title="Data",
            yaxis_title="Quantidade de Alertas",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig_timeline, use_container_width=True, key='individual_line_chart')
    
        tab1, tab2, tab3 = st.tabs(["🔴 Ocorrências Isoladas", "🟢 Ocorrências Agrupadas", "📊 Visualização Temporal"])
    
        with tab1:
            st.subheader(f"🔴 Ocorrências Isoladas ({len(df_isolated)})")
            if len(df_isolated) > 0:
                isolated_display = df_isolated[['created_on', 'hour', 'day_name', 'time_diff_hours', 'date']].copy()
                isolated_display['created_on'] = isolated_display['created_on'].dt.strftime('%Y-%m-%d %H:%M:%S')
                isolated_display.columns = ['Data/Hora', 'Hora', 'Dia da Semana', 'Intervalo (h)', 'Data']
                st.dataframe(isolated_display, use_container_width=True)
                st.write(f"**Percentual:** {len(df_isolated)/len(self.df)*100:.2f}% das ocorrências são isoladas")
                
                daily_counts = df_isolated.groupby('date').size().sort_values(ascending=False)
                if len(daily_counts) > 0:
                    st.write("**📈 Dias com Mais Alertas Isolados:**")
                    top_days = daily_counts.head(5)
                    for date, count in top_days.items():
                        st.write(f"• {date}: {count} alertas")
            else:
                st.info("Nenhuma ocorrência isolada detectada neste alerta.")
    
        with tab2:
            st.subheader(f"🟢 Ocorrências Agrupadas ({len(df_grouped)})")
            if len(df_grouped) > 0:
                grouped_display = df_grouped[['created_on', 'hour', 'day_name', 'time_diff_hours', 'group_id']].copy()
                grouped_display['created_on'] = grouped_display['created_on'].dt.strftime('%Y-%m-%d %H:%M:%S')
                grouped_display.columns = ['Data/Hora', 'Hora', 'Dia da Semana', 'Intervalo (h)', 'Grupo']
                st.dataframe(grouped_display, use_container_width=True)
                st.write(f"**Percentual:** {len(df_grouped)/len(self.df)*100:.2f}% das ocorrências estão agrupadas")
            else:
                st.info("Nenhuma ocorrência agrupada detectada neste alerta.")
        
        with tab3:
            st.subheader("📊 Visualização Temporal dos Alertas")
            
            fig = go.Figure()
            
            if len(df_isolated) > 0:
                fig.add_trace(go.Scatter(
                    x=df_isolated['created_on'],
                    y=[1] * len(df_isolated),
                    mode='markers',
                    name='Isolados',
                    marker=dict(size=10, color='red', symbol='x'),
                    hovertemplate='%{x}<br>Isolado<extra></extra>'
                ))
            
            for group_info in self.groups_info:
                group_id = group_info['group_id']
                group_data = df_grouped[df_grouped['group_id'] == group_id]
                fig.add_trace(go.Scatter(
                    x=group_data['created_on'],
                    y=[1] * len(group_data),
                    mode='markers',
                    name=f'Grupo {group_id}',
                    marker=dict(size=10),
                    hovertemplate='%{x}<br>Grupo ' + str(group_id) + '<extra></extra>'
                ))
            
            fig.update_layout(
                title="Timeline de Alertas (Isolados vs Agrupados)",
                xaxis_title="Data/Hora",
                yaxis=dict(showticklabels=False, title=""),
                height=400,
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True, key='individual_alert_timeline')
            

def main():
    st.title("🚨 Analisador de Alertas")
    st.markdown("### Análise individual, global e agrupamento inteligente de alertas")
    st.sidebar.header("⚙️ Configurações")
    
    with st.sidebar.expander("🎛️ Parâmetros de Agrupamento", expanded=False):
        max_gap_hours = st.slider(
            "⏱️ Gap Máximo Entre Alertas (horas)",
            min_value=1,
            max_value=72,
            value=24,
            help="Alertas separados por mais tempo que isso são considerados de grupos diferentes"
        )
        min_group_size = st.slider(
            "📊 Tamanho Mínimo do Grupo",
            min_value=2,
            max_value=10,
            value=3,
            help="Número mínimo de alertas para formar um grupo válido"
        )
        spike_threshold_multiplier = st.slider(
            "🚀 Multiplicador de Spike",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="Dias com mais alertas que média × este valor são considerados spikes isolados"
        )
    
    analysis_mode = st.sidebar.selectbox(
        "🎯 Modo de Análise",
        ["🌍 Análise Global", "🔍 Análise Individual"],
        help="Escolha entre analisar todos os alertas ou um alerta específico"
    )
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload do arquivo CSV",
        type=['csv'],
        help="Faça upload do arquivo CSV contendo os dados dos alertas"
    )
    
    if uploaded_file is not None:
        analyzer = StreamlitAlertAnalyzer()
        if analyzer.load_data(uploaded_file):
            if analysis_mode == "🌍 Análise Global":
                st.markdown("---")
                use_multiprocessing = st.sidebar.checkbox(
                    "⚡ Usar Multiprocessing (Mais Rápido)", 
                    value=True,
                    help="Processa alertas em paralelo para melhor desempenho"
                )
                if st.sidebar.button("🚀 Executar Análise Global", type="primary"):
                    if analyzer.prepare_global_analysis(use_multiprocessing, max_gap_hours, 
                                                       min_group_size, spike_threshold_multiplier):
                        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                            "📊 Visão Geral",
                            "🔍 Isolados vs Contínuos",
                            "🔬 Grupos Detalhados",
                            "🔁 Recorrência",
                            "🎯 Agrupamento", 
                            "👥 Perfis dos Clusters",
                            "💡 Recomendações"
                        ])
                        with tab1:
                            analyzer.show_global_overview()
                        with tab2:
                            analyzer.show_isolated_vs_continuous_analysis()
                        with tab3:
                            analyzer.show_continuous_groups_detailed_view()
                        with tab4:
                            analyzer.analyze_continuous_recurrence_patterns()
                        with tab5:
                            n_clusters = analyzer.perform_clustering_analysis()
                        with tab6:
                            if n_clusters:
                                analyzer.show_cluster_profiles(n_clusters)
                        with tab7:
                            if n_clusters:
                                analyzer.show_cluster_recommendations()
                        st.sidebar.markdown("---")
                        st.sidebar.subheader("📥 Downloads")
                        csv_buffer = io.StringIO()
                        analyzer.df_all_alerts.to_csv(csv_buffer, index=False)
                        st.sidebar.download_button(
                            label="⬇️ Baixar Análise Global",
                            data=csv_buffer.getvalue(),
                            file_name=f"analise_global_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("❌ Não foi possível processar os dados para análise global")
            else:
                try:
                    id_counts = analyzer.df_original['u_alert_id'].value_counts()
                    id_options = [f"{uid} ({count} ocorrências)" for uid, count in id_counts.items()]
                    selected_option = st.sidebar.selectbox(
                        "🎯 Selecione o Alert ID",
                        id_options,
                        help="Escolha o ID do alerta para análise (ordenado por frequência)"
                    )
                    selected_id = selected_option.split(" ")[0]

                    if st.sidebar.button("🚀 Executar Análise Individual", type="primary"):
                        analyzer.max_gap_hours = max_gap_hours
                        analyzer.min_group_size = min_group_size
                        analyzer.spike_threshold_multiplier = spike_threshold_multiplier

                        if analyzer.prepare_individual_analysis(selected_id):
                            st.success(f"🎯 Analisando alert_id: {selected_id} ({len(analyzer.df)} registros)")
                            st.info(f"📅 **Período analisado:** {analyzer.dates.min()} até {analyzer.dates.max()}")

                            tab1, tab2, tab3 = st.tabs([
                                "🔍 Isolados vs Agrupados",
                                "📊 Básico", 
                                "⏱️ Análise Avançada de Reincidência"
                            ])

                            with tab1:
                                analyzer.show_individual_alert_analysis()
                            with tab2:
                                analyzer.show_basic_stats()
                            with tab3:
                                analyzer.analyze_temporal_recurrence_patterns()

                            st.sidebar.markdown("---")
                            st.sidebar.subheader("📥 Download")

                            csv_buffer = io.StringIO()
                            analyzer.df.to_csv(csv_buffer, index=False)
                            st.sidebar.download_button(
                                label="⬇️ Baixar Dados Processados",
                                data=csv_buffer.getvalue(),
                                file_name=f"analise_{selected_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(f"❌ Nenhum registro encontrado para alert_id: {selected_id}")
                except Exception as e:
                    st.error(f"❌ Erro ao processar análise individual: {e}")
    else:
        st.info("👆 Faça upload de um arquivo CSV para começar a análise")
        with st.expander("📖 Instruções de Uso"):
            st.markdown("""
            ### Como usar este analisador:
            
            #### 🌍 **Análise Global**
            Analise todos os alertas com 7 abas:
            1. **Visão Geral:** Top alertas e distribuições
            2. **Isolados vs Contínuos:** Comparação detalhada com gráfico temporal
            3. **Grupos Detalhados:** Visualização interativa dos grupos identificados em alertas contínuos
            4. **Recorrência:** Padrões de hora/dia APENAS de alertas contínuos
            5. **Agrupamento:** Clustering por comportamento
            6. **Perfis:** Características de cada cluster
            7. **Recomendações:** Ações sugeridas
            
            #### 🔍 **Análise Individual**
            Analise um alerta específico em 3 abas:
            1. **Isolados vs Agrupados:** Classificação e timeline
            2. **Básico:** Estatísticas gerais
            3. **Análise Avançada de Reincidência:** 14 análises completas incluindo:
               - Estatísticas de intervalos
               - Classificação de regularidade
               - Periodicidade (FFT)
               - Autocorrelação
               - Padrões temporais
               - Clusters temporais
               - Detecção de bursts
               - Sazonalidade
               - Pontos de mudança
               - Detecção de anomalias
               - Análise de tendência
               - Score de previsibilidade
               - Análise de estabilidade
               - Pattern mining
               - **Classificação final de reincidência com score 0-100**
            
            ### Principais Funcionalidades:
            - ✨ Identificação automática de grupos contínuos
            - 📊 Visualização detalhada de grupos com timeline
            - 📈 Análise de recorrência (hora/dia) para alertas contínuos
            - 🎯 Clustering inteligente por perfil de comportamento
            - ⏱️ **14 análises avançadas de reincidência com ML**
            - 🔴 Separação clara entre alertas isolados e contínuos
            - 🏆 **Score final de reincidência (0-100) com priorização**
            - 🔮 Predição de próxima ocorrência
            - 📥 Exportação de relatórios completos
            
            ### Colunas necessárias no CSV:
            - `u_alert_id`: Identificador único do alerta
            - `created_on`: Data e hora da criação do alerta
            
            ### Parâmetros Configuráveis:
            - **Gap Máximo:** Tempo máximo entre alertas do mesmo grupo
            - **Tamanho Mínimo:** Quantidade mínima de alertas para formar um grupo
            - **Multiplicador de Spike:** Threshold para identificar dias com picos anormais
            """)

if __name__ == "__main__":
    main()
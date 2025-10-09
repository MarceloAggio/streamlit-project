import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Analisador de Alertas - Completo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FUNÇÕES AUXILIARES DE AGRUPAMENTO
# ============================================================

def identify_alert_groups(alert_data, max_gap_hours=24, min_group_size=3, 
                         spike_threshold_multiplier=5):
    """
    Identifica grupos/sessões de alertas baseado em intervalos de tempo.
    Alertas isolados são aqueles que não pertencem a nenhum grupo significativo.
    """
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
    """
    Classifica um alerta baseado na identificação de grupos.
    """
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
# FUNÇÕES DE PROCESSAMENTO
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
    # ANÁLISE GLOBAL - ISOLADOS VS CONTÍNUOS
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

    # ============================================================
    # VISUALIZAÇÃO DETALHADA DOS GRUPOS CONTÍNUOS
    # ============================================================

    def show_continuous_groups_detailed_view(self):
        """
        Mostra visualização detalhada dos grupos identificados nos alertas contínuos
        """
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

    # ============================================================
    # ANÁLISE DE RECORRÊNCIA - ALERTAS CONTÍNUOS
    # ============================================================

    def analyze_continuous_recurrence_patterns(self):
        """
        Analisa padrões de recorrência APENAS dos alertas contínuos.
        """
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
        
        st.subheader("🎯 Resumo do Padrão de Recorrência")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**⏰ Padrão Horário:**")
            st.write(f"• {pattern_hour}")
            st.write(f"• Top 3 horas concentram {total_top_3_hours:.1f}% dos alertas")
            st.write(f"• Horário principal: **{top_3_hours.index[0]:02d}:00**")
            
            if total_top_3_hours > 50:
                st.write("• 💡 **Recomendação:** Considerar janela de manutenção específica")
        
        with col2:
            st.write("**📅 Padrão Semanal:**")
            st.write(f"• {pattern_day}")
            st.write(f"• Top 3 dias concentram {total_top_3_days:.1f}% dos alertas")
            st.write(f"• Dia principal: **{day_translation[top_day_en]}**")
            
            if total_top_3_days > 50:
                st.write("• 💡 **Recomendação:** Atenção redobrada nestes dias")
        
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

    # ============================================================
    # CLUSTERING
    # ============================================================

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

    # ============================================================
    # ANÁLISE INDIVIDUAL
    # ============================================================

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

    def analyze_temporal_recurrence_patterns(self):
    
        st.header("🔄 Análise Avançada de Recorrência Temporal")
        
        if self.df is None or len(self.df) < 3:
            st.warning("⚠️ Dados insuficientes para análise de recorrência (mínimo 3 ocorrências).")
            return
        
        # Preparar dados temporais
        df_sorted = self.df.sort_values('created_on').copy()
        st.info(f"📊 Analisando padrões de recorrência para **{len(df_sorted)}** ocorrências do Alert ID: **{self.alert_id}**")
        
        # Calcular intervalos
        df_sorted['timestamp'] = df_sorted['created_on'].astype('int64') // 10**9
        df_sorted['time_diff_seconds'] = df_sorted['timestamp'].diff()
        df_sorted['time_diff_hours'] = df_sorted['time_diff_seconds'] / 3600
        df_sorted['time_diff_days'] = df_sorted['time_diff_seconds'] / 86400
        
        intervals_seconds = df_sorted['time_diff_seconds'].dropna().values
        intervals_hours = df_sorted['time_diff_hours'].dropna().values
        
        if len(intervals_seconds) < 2:
            st.warning("⚠️ Intervalos insuficientes para análise completa de recorrência.")
            return
        
        # ============================================================
        # 1. ESTATÍSTICAS BÁSICAS DE INTERVALO
        # ============================================================
        st.subheader("📊 1. Estatísticas de Intervalos")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("⏱️ Média", f"{np.mean(intervals_hours):.1f}h")
        with col2:
            st.metric("📊 Mediana", f"{np.median(intervals_hours):.1f}h")
        with col3:
            st.metric("📈 Desvio Padrão", f"{np.std(intervals_hours):.1f}h")
        with col4:
            st.metric("⚡ Mínimo", f"{np.min(intervals_hours):.1f}h")
        with col5:
            st.metric("🐌 Máximo", f"{np.max(intervals_hours):.1f}h")
        
        # Coeficiente de variação para determinar regularidade
        cv = np.std(intervals_hours) / np.mean(intervals_hours) if np.mean(intervals_hours) > 0 else float('inf')
        
        # ============================================================
        # 2. CLASSIFICAÇÃO DE REGULARIDADE
        # ============================================================
        st.subheader("🎯 2. Classificação de Regularidade")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if cv < 0.15:
                pattern_type = "🟢 **ALTAMENTE REGULAR**"
                pattern_desc = "Intervalos extremamente consistentes - possível processo automatizado"
                pattern_color = "green"
                regularity_score = 95
            elif cv < 0.35:
                pattern_type = "🟢 **REGULAR**"
                pattern_desc = "Intervalos consistentes com pequenas variações"
                pattern_color = "lightgreen"
                regularity_score = 80
            elif cv < 0.65:
                pattern_type = "🟡 **SEMI-REGULAR**"
                pattern_desc = "Padrão detectável mas com variações moderadas"
                pattern_color = "yellow"
                regularity_score = 60
            elif cv < 1.0:
                pattern_type = "🟠 **IRREGULAR**"
                pattern_desc = "Intervalos inconsistentes - possível múltiplas causas"
                pattern_color = "orange"
                regularity_score = 40
            else:
                pattern_type = "🔴 **ALTAMENTE IRREGULAR**"
                pattern_desc = "Sem padrão detectável - comportamento caótico ou aleatório"
                pattern_color = "red"
                regularity_score = 20
            
            st.markdown(f"**Classificação:** {pattern_type}")
            st.write(pattern_desc)
            st.write(f"**📊 Coeficiente de Variação:** {cv:.2%}")
            
            # Teste de aleatoriedade usando runs test
            median_val = np.median(intervals_hours)
            runs = []
            current_run = []
            
            for val in intervals_hours:
                if len(current_run) == 0:
                    current_run.append(val > median_val)
                elif (val > median_val) == current_run[-1]:
                    current_run.append(val > median_val)
                else:
                    runs.append(len(current_run))
                    current_run = [val > median_val]
            if current_run:
                runs.append(len(current_run))
            
            num_runs = len(runs)
            expected_runs = (2 * len(intervals_hours) / 3) + 1
            
            if abs(num_runs - expected_runs) / expected_runs < 0.2:
                st.info("📊 **Teste de Aleatoriedade:** Padrão consistente com comportamento aleatório")
            else:
                st.success("✅ **Teste de Aleatoriedade:** Padrão NÃO aleatório detectado - possível recorrência")
        
        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = regularity_score,
                title = {'text': "Score de Regularidade"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': pattern_color},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgray"},
                        {'range': [20, 40], 'color': "lightyellow"},
                        {'range': [40, 60], 'color': "lightgreen"},
                        {'range': [60, 80], 'color': "green"},
                        {'range': [80, 100], 'color': "darkgreen"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True, key='regularity_gauge_advanced')
        
        # ============================================================
        # 3. DETECÇÃO DE PERIODICIDADE (FFT)
        # ============================================================
        st.subheader("🔍 3. Análise de Periodicidade (FFT)")
        
        if len(intervals_hours) >= 10:
            # Aplicar FFT para detectar frequências dominantes
            from scipy.fft import fft, fftfreq
            
            # Normalizar e aplicar FFT
            intervals_normalized = (intervals_hours - np.mean(intervals_hours)) / np.std(intervals_hours)
            
            # Padding para melhorar FFT
            n = len(intervals_normalized)
            n_padded = 2**int(np.ceil(np.log2(n)))
            intervals_padded = np.pad(intervals_normalized, (0, n_padded - n), 'constant')
            
            fft_values = fft(intervals_padded)
            frequencies = fftfreq(n_padded, d=1)
            
            # Pegar apenas frequências positivas
            positive_freq_idx = frequencies > 0
            frequencies_positive = frequencies[positive_freq_idx]
            fft_magnitude = np.abs(fft_values[positive_freq_idx])
            
            # Encontrar picos de frequência
            threshold = np.mean(fft_magnitude) + 2 * np.std(fft_magnitude)
            peaks_idx = fft_magnitude > threshold
            
            if np.any(peaks_idx):
                dominant_frequencies = frequencies_positive[peaks_idx]
                dominant_periods = 1 / dominant_frequencies
                
                st.success("🎯 **Periodicidades Detectadas:**")
                for i, period in enumerate(dominant_periods[:3]):  # Top 3 períodos
                    if period < len(intervals_hours):  # Filtrar períodos muito longos
                        st.write(f"• Período de aproximadamente **{period:.1f}** ocorrências")
                        estimated_time = period * np.mean(intervals_hours)
                        if estimated_time < 24:
                            st.write(f"  → Equivale a ~**{estimated_time:.1f} horas**")
                        else:
                            st.write(f"  → Equivale a ~**{estimated_time/24:.1f} dias**")
            else:
                st.info("📊 Nenhuma periodicidade forte detectada via FFT")
            
            # Visualização FFT
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=1/frequencies_positive[:len(frequencies_positive)//4],  # Converter para período
                y=fft_magnitude[:len(frequencies_positive)//4],
                mode='lines',
                name='Magnitude FFT'
            ))
            fig_fft.update_layout(
                title="Espectro de Frequência (FFT)",
                xaxis_title="Período (número de ocorrências)",
                yaxis_title="Magnitude",
                xaxis_type="log",
                height=350
            )
            st.plotly_chart(fig_fft, use_container_width=True, key='fft_plot')
        else:
            st.info("📊 Mínimo de 10 intervalos necessários para análise FFT")
        
        # ============================================================
        # 4. AUTOCORRELAÇÃO
        # ============================================================
        st.subheader("📈 4. Análise de Autocorrelação")
        
        if len(intervals_hours) >= 5:
            from scipy import signal
            
            # Calcular autocorrelação
            intervals_normalized = (intervals_hours - np.mean(intervals_hours)) / np.std(intervals_hours)
            autocorr = signal.correlate(intervals_normalized, intervals_normalized, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Pegar apenas metade positiva
            autocorr = autocorr / autocorr[0]  # Normalizar
            
            # Encontrar picos significativos
            lags = np.arange(len(autocorr))
            significant_threshold = 2 / np.sqrt(len(intervals_hours))  # 95% confidence
            
            # Encontrar primeiro pico significativo após lag 0
            significant_peaks = []
            for i in range(1, min(len(autocorr), 20)):
                if autocorr[i] > significant_threshold:
                    significant_peaks.append((i, autocorr[i]))
            
            if significant_peaks:
                st.success("✅ **Autocorrelação Significativa Detectada:**")
                for lag, corr_value in significant_peaks[:3]:
                    st.write(f"• Lag {lag}: correlação de {corr_value:.2f}")
                    st.write(f"  → Sugere repetição a cada ~{lag} ocorrências")
            else:
                st.info("📊 Sem autocorrelação significativa - padrão não repetitivo")
            
            # Visualização
            fig_autocorr = go.Figure()
            fig_autocorr.add_trace(go.Scatter(
                x=lags[:min(30, len(lags))],
                y=autocorr[:min(30, len(autocorr))],
                mode='lines+markers',
                name='Autocorrelação'
            ))
            fig_autocorr.add_hline(
                y=significant_threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Threshold 95%"
            )
            fig_autocorr.add_hline(
                y=-significant_threshold, 
                line_dash="dash", 
                line_color="red"
            )
            fig_autocorr.update_layout(
                title="Função de Autocorrelação",
                xaxis_title="Lag",
                yaxis_title="Correlação",
                height=350
            )
            st.plotly_chart(fig_autocorr, use_container_width=True, key='autocorr_plot')
        
        # ============================================================
        # 5. ANÁLISE DE PADRÕES TEMPORAIS
        # ============================================================
        st.subheader("⏰ 5. Padrões Temporais Recorrentes")
        
        # Análise por hora do dia
        hourly_pattern = df_sorted.groupby('hour').size()
        hourly_pattern = hourly_pattern.reindex(range(24), fill_value=0)
        
        # Análise por dia da semana
        daily_pattern = df_sorted.groupby('day_of_week').size()
        daily_pattern = daily_pattern.reindex(range(7), fill_value=0)
        
        # Análise por dia do mês
        df_sorted['day_of_month'] = df_sorted['created_on'].dt.day
        monthly_pattern = df_sorted.groupby('day_of_month').size()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Padrão horário
            fig_hour = go.Figure()
            fig_hour.add_trace(go.Bar(
                x=list(range(24)),
                y=hourly_pattern.values,
                marker_color=['red' if v > hourly_pattern.mean() + hourly_pattern.std() else 'lightblue' 
                            for v in hourly_pattern.values]
            ))
            fig_hour.update_layout(
                title="Padrão de Recorrência por Hora",
                xaxis_title="Hora do Dia",
                yaxis_title="Ocorrências",
                height=300
            )
            st.plotly_chart(fig_hour, use_container_width=True, key='hourly_pattern')
            
            # Detectar janelas horárias
            peak_hours = hourly_pattern[hourly_pattern > hourly_pattern.mean() + hourly_pattern.std()].index.tolist()
            if peak_hours:
                st.success(f"🕐 **Horas de pico:** {', '.join([f'{h:02d}:00' for h in peak_hours])}")
        
        with col2:
            # Padrão semanal
            days_map = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
            fig_day = go.Figure()
            fig_day.add_trace(go.Bar(
                x=days_map,
                y=daily_pattern.values,
                marker_color=['red' if v > daily_pattern.mean() + daily_pattern.std() else 'lightgreen' 
                            for v in daily_pattern.values]
            ))
            fig_day.update_layout(
                title="Padrão de Recorrência por Dia",
                xaxis_title="Dia da Semana",
                yaxis_title="Ocorrências",
                height=300
            )
            st.plotly_chart(fig_day, use_container_width=True, key='daily_pattern')
            
            # Detectar dias recorrentes
            peak_days = daily_pattern[daily_pattern > daily_pattern.mean() + daily_pattern.std()].index.tolist()
            if peak_days:
                st.success(f"📅 **Dias de pico:** {', '.join([days_map[d] for d in peak_days])}")
        
        # ============================================================
        # 6. DETECÇÃO DE CLUSTERS TEMPORAIS
        # ============================================================
        st.subheader("🎯 6. Detecção de Clusters Temporais")
        
        if len(df_sorted) >= 10:
            # Usar DBSCAN para encontrar clusters temporais
            from sklearn.cluster import DBSCAN
            
            # Preparar dados para clustering (timestamp em horas desde o início)
            first_timestamp = df_sorted['timestamp'].min()
            time_features = ((df_sorted['timestamp'] - first_timestamp) / 3600).values.reshape(-1, 1)
            
            # Determinar eps baseado na mediana dos intervalos
            eps_value = np.median(intervals_hours) * 2
            
            # Aplicar DBSCAN
            dbscan = DBSCAN(eps=eps_value, min_samples=3)
            clusters = dbscan.fit_predict(time_features)
            
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Clusters Detectados", n_clusters)
            with col2:
                st.metric("📊 Alertas em Clusters", len(clusters) - n_noise)
            with col3:
                st.metric("🔴 Alertas Isolados", n_noise)
            
            if n_clusters > 0:
                st.success(f"✅ Identificados **{n_clusters} clusters temporais** distintos")
                
                # Análise de cada cluster
                cluster_info = []
                for cluster_id in set(clusters):
                    if cluster_id != -1:
                        cluster_mask = clusters == cluster_id
                        cluster_times = df_sorted[cluster_mask]['created_on']
                        cluster_info.append({
                            'Cluster': cluster_id,
                            'Tamanho': cluster_mask.sum(),
                            'Início': cluster_times.min().strftime('%Y-%m-%d %H:%M'),
                            'Fim': cluster_times.max().strftime('%Y-%m-%d %H:%M'),
                            'Duração (h)': (cluster_times.max() - cluster_times.min()).total_seconds() / 3600
                        })
                
                if cluster_info:
                    cluster_df = pd.DataFrame(cluster_info)
                    st.dataframe(cluster_df, use_container_width=True)
                    
                    # Calcular intervalo entre clusters
                    if len(cluster_info) > 1:
                        inter_cluster_intervals = []
                        for i in range(len(cluster_info) - 1):
                            end_current = pd.to_datetime(cluster_info[i]['Fim'])
                            start_next = pd.to_datetime(cluster_info[i+1]['Início'])
                            interval_hours = (start_next - end_current).total_seconds() / 3600
                            inter_cluster_intervals.append(interval_hours)
                        
                        avg_inter_cluster = np.mean(inter_cluster_intervals)
                        std_inter_cluster = np.std(inter_cluster_intervals)
                        
                        st.info(f"📊 **Intervalo médio entre clusters:** {avg_inter_cluster:.1f}h ± {std_inter_cluster:.1f}h")
                        
                        if std_inter_cluster / avg_inter_cluster < 0.3:
                            st.success("✅ **Clusters aparecem em intervalos regulares** - forte indício de recorrência")
        
        # ============================================================
        # 7. RESUMO E DIAGNÓSTICO FINAL
        # ============================================================
        st.subheader("📋 7. Diagnóstico Final de Recorrência")
        
        # Calcular score final de recorrência
        recurrence_indicators = []
        recurrence_score = 0
        
        # Indicador 1: Regularidade dos intervalos
        if cv < 0.5:
            recurrence_indicators.append("✅ Intervalos regulares")
            recurrence_score += 25
        else:
            recurrence_indicators.append("❌ Intervalos irregulares")
        
        # Indicador 2: Periodicidade detectada
        if 'dominant_periods' in locals() and len(dominant_periods) > 0:
            recurrence_indicators.append("✅ Periodicidade detectada via FFT")
            recurrence_score += 25
        else:
            recurrence_indicators.append("❌ Sem periodicidade clara")
        
        # Indicador 3: Autocorrelação significativa
        if 'significant_peaks' in locals() and significant_peaks:
            recurrence_indicators.append("✅ Autocorrelação significativa")
            recurrence_score += 25
        else:
            recurrence_indicators.append("❌ Sem autocorrelação")
        
        # Indicador 4: Clusters temporais regulares
        if 'n_clusters' in locals() and n_clusters > 1:
            recurrence_indicators.append("✅ Clusters temporais identificados")
            recurrence_score += 25
        else:
            recurrence_indicators.append("❌ Sem clusters temporais")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**📊 Indicadores de Recorrência:**")
            for indicator in recurrence_indicators:
                st.write(f"• {indicator}")
            
            st.write("\n**🎯 Diagnóstico:**")
            if recurrence_score >= 75:
                st.success("**ALTA RECORRÊNCIA** - Padrão altamente previsível")
                st.write("💡 **Recomendação:** Ideal para automação e agendamento preventivo")
            elif recurrence_score >= 50:
                st.warning("**RECORRÊNCIA MODERADA** - Padrão parcialmente previsível")
                st.write("💡 **Recomendação:** Monitorar tendências e considerar automação parcial")
            elif recurrence_score >= 25:
                st.info("**BAIXA RECORRÊNCIA** - Padrão pouco previsível")
                st.write("💡 **Recomendação:** Investigar causas múltiplas e variáveis")
            else:
                st.error("**SEM RECORRÊNCIA** - Comportamento aleatório")
                st.write("💡 **Recomendação:** Análise caso a caso e investigação de causas raiz")
        
        with col2:
            fig_score = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = recurrence_score,
                title = {'text': "Score de Recorrência"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightblue"},
                        {'range': [75, 100], 'color': "blue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            fig_score.update_layout(height=300)
            st.plotly_chart(fig_score, use_container_width=True, key='recurrence_score')
        
        # ============================================================
        # 8. PREDIÇÃO DE PRÓXIMA OCORRÊNCIA
        # ============================================================
        if recurrence_score >= 50 and len(intervals_hours) >= 3:
            st.subheader("🔮 8. Predição de Próxima Ocorrência")
            
            last_alert_time = df_sorted['created_on'].max()
            
            # Método 1: Baseado na média
            pred_mean = last_alert_time + pd.Timedelta(hours=np.mean(intervals_hours))
            
            # Método 2: Baseado na mediana
            pred_median = last_alert_time + pd.Timedelta(hours=np.median(intervals_hours))
            
            # Método 3: Baseado no último intervalo
            pred_last = last_alert_time + pd.Timedelta(hours=intervals_hours[-1])
            
            # Intervalo de confiança
            confidence_interval = 1.96 * np.std(intervals_hours) / np.sqrt(len(intervals_hours))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Predição (Média)", pred_mean.strftime('%d/%m %H:%M'))
            with col2:
                st.metric("📊 Predição (Mediana)", pred_median.strftime('%d/%m %H:%M'))
            with col3:
                st.metric("📊 Predição (Último)", pred_last.strftime('%d/%m %H:%M'))
            
            st.info(f"📈 **Intervalo de Confiança (95%):** ± {confidence_interval:.1f} horas")
            
            # Se houver padrão horário forte, ajustar predição
            if peak_hours:
                st.write(f"💡 **Ajuste sugerido:** Considerar horários de pico às {', '.join([f'{h:02d}:00' for h in peak_hours[:3]])}")

        
        st.subheader("💥 9. Detecção de Bursts (Rajadas)")
        
        # Detectar bursts usando método de Kleinberg
        burst_threshold = np.percentile(intervals_hours, 25)  # Quartil inferior
        
        # Identificar sequências de intervalos curtos
        burst_sequences = []
        current_burst = []
        
        for i, interval in enumerate(intervals_hours):
            if interval < burst_threshold:
                if not current_burst:
                    current_burst = [i]
                current_burst.append(i + 1)
            else:
                if len(current_burst) >= 3:  # Mínimo de 3 alertas para considerar burst
                    burst_sequences.append(current_burst)
                current_burst = []
        
        if len(current_burst) >= 3:
            burst_sequences.append(current_burst)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💥 Bursts Detectados", len(burst_sequences))
        
        with col2:
            if burst_sequences:
                avg_burst_size = np.mean([len(b) for b in burst_sequences])
                st.metric("📊 Tamanho Médio", f"{avg_burst_size:.1f} alertas")
            else:
                st.metric("📊 Tamanho Médio", "N/A")
        
        with col3:
            if burst_sequences:
                max_burst_size = max([len(b) for b in burst_sequences])
                st.metric("🔥 Maior Burst", f"{max_burst_size} alertas")
            else:
                st.metric("🔥 Maior Burst", "N/A")
        
        if burst_sequences:
            st.warning(f"⚠️ **Padrão de Rajadas Detectado:** {len(burst_sequences)} bursts identificados")
            
            # Análise temporal dos bursts
            burst_times = []
            for burst in burst_sequences:
                burst_start_idx = burst[0]
                if burst_start_idx < len(df_sorted) - 1:
                    burst_time = df_sorted.iloc[burst_start_idx]['created_on']
                    burst_times.append(burst_time)
            
            if len(burst_times) > 1:
                burst_df = pd.DataFrame({'burst_time': burst_times})
                burst_df['hour'] = burst_df['burst_time'].dt.hour
                burst_df['day_of_week'] = burst_df['burst_time'].dt.dayofweek
                
                burst_hour_pattern = burst_df['hour'].value_counts().head(3)
                if not burst_hour_pattern.empty:
                    st.info(f"🕐 **Horários com mais bursts:** {', '.join([f'{h:02d}:00' for h in burst_hour_pattern.index])}")
        else:
            st.success("✅ Sem padrão de rajadas - distribuição uniforme")
        
        # ============================================================
        # 10. ANÁLISE DE SAZONALIDADE AVANÇADA
        # ============================================================
        st.subheader("🌡️ 10. Análise de Sazonalidade")
        
        # Verificar se temos dados suficientes para análise sazonal
        date_range = (df_sorted['created_on'].max() - df_sorted['created_on'].min()).days
        
        if date_range >= 30:  # Pelo menos 30 dias de dados
            # Análise mensal
            df_sorted['month'] = df_sorted['created_on'].dt.month
            df_sorted['week_of_year'] = df_sorted['created_on'].dt.isocalendar().week
            
            col1, col2 = st.columns(2)
            
            with col1:
                if date_range >= 90:  # 3+ meses para análise mensal
                    monthly_pattern = df_sorted.groupby('month').size()
                    
                    fig_month = go.Figure()
                    months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                            'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                    
                    month_values = [monthly_pattern.get(i+1, 0) for i in range(12)]
                    fig_month.add_trace(go.Bar(
                        x=months,
                        y=month_values,
                        marker_color=['red' if v > np.mean([x for x in month_values if x > 0]) * 1.5 else 'lightblue' 
                                    for v in month_values]
                    ))
                    fig_month.update_layout(
                        title="Padrão Mensal",
                        xaxis_title="Mês",
                        yaxis_title="Ocorrências",
                        height=300
                    )
                    st.plotly_chart(fig_month, use_container_width=True, key='monthly_pattern')
                    
                    # Detectar meses anômalos
                    active_months = [i for i, v in enumerate(month_values) if v > 0]
                    if active_months and len(active_months) >= 3:
                        active_values = [month_values[i] for i in active_months]
                        threshold = np.mean(active_values) + 1.5 * np.std(active_values)
                        anomaly_months = [months[i] for i, v in enumerate(month_values) if v > threshold]
                        if anomaly_months:
                            st.warning(f"📅 **Meses anômalos:** {', '.join(anomaly_months)}")
            
            with col2:
                # Análise por semana do ano
                weekly_pattern = df_sorted.groupby('week_of_year').size()
                
                if len(weekly_pattern) >= 4:
                    fig_week = go.Figure()
                    fig_week.add_trace(go.Scatter(
                        x=weekly_pattern.index,
                        y=weekly_pattern.values,
                        mode='lines+markers',
                        fill='tozeroy'
                    ))
                    fig_week.update_layout(
                        title="Padrão por Semana do Ano",
                        xaxis_title="Semana",
                        yaxis_title="Ocorrências",
                        height=300
                    )
                    st.plotly_chart(fig_week, use_container_width=True, key='weekly_pattern')
                    
                    # Calcular tendência
                    from scipy import stats as scipy_stats
                    weeks = weekly_pattern.index.values
                    counts = weekly_pattern.values
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(weeks, counts)
                    
                    if p_value < 0.05:
                        if slope > 0:
                            st.warning("📈 **Tendência crescente** detectada ao longo do tempo")
                        else:
                            st.success("📉 **Tendência decrescente** detectada ao longo do tempo")
                    else:
                        st.info("➡️ **Sem tendência significativa** ao longo do tempo")
        
        # ============================================================
        # 11. ANÁLISE DE ENTROPIA E COMPLEXIDADE
        # ============================================================
        st.subheader("🧬 11. Análise de Entropia e Complexidade")
        
        # Calcular entropia de Shannon dos intervalos
        if len(intervals_hours) >= 10:
            # Discretizar intervalos em bins
            n_bins = min(10, len(intervals_hours) // 3)
            hist, bin_edges = np.histogram(intervals_hours, bins=n_bins)
            
            # Calcular probabilidades
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remover zeros
            
            # Entropia de Shannon
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(n_bins)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Entropia", f"{entropy:.2f} bits")
            
            with col2:
                st.metric("📈 Entropia Normalizada", f"{normalized_entropy:.2%}")
            
            with col3:
                # Classificação baseada em entropia
                if normalized_entropy < 0.3:
                    complexity = "Muito Baixa"
                    complexity_color = "🟢"
                    complexity_desc = "Padrão muito previsível"
                elif normalized_entropy < 0.5:
                    complexity = "Baixa"
                    complexity_color = "🟢"
                    complexity_desc = "Padrão previsível"
                elif normalized_entropy < 0.7:
                    complexity = "Média"
                    complexity_color = "🟡"
                    complexity_desc = "Complexidade moderada"
                elif normalized_entropy < 0.85:
                    complexity = "Alta"
                    complexity_color = "🟠"
                    complexity_desc = "Padrão complexo"
                else:
                    complexity = "Muito Alta"
                    complexity_color = "🔴"
                    complexity_desc = "Comportamento caótico"
                
                st.metric("🧬 Complexidade", f"{complexity_color} {complexity}")
            
            st.info(f"💡 **Interpretação:** {complexity_desc}")
            
            # Sample Entropy (medida de regularidade)
            def sample_entropy(data, m=2, r=0.2):
                """Calcula Sample Entropy - medida de irregularidade"""
                N = len(data)
                if N < m + 1:
                    return float('nan')
                
                def _maxdist(x_i, x_j):
                    return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
                
                def _phi(m):
                    patterns = np.array([data[i:i+m] for i in range(N - m + 1)])
                    C = 0
                    for i in range(N - m + 1):
                        template = patterns[i]
                        matches = 0
                        for j in range(N - m + 1):
                            if i != j and _maxdist(template, patterns[j]) <= r * np.std(data):
                                matches += 1
                        if matches > 0:
                            C += matches / (N - m)
                    return C / (N - m + 1) if (N - m + 1) > 0 else 0
                
                phi_m = _phi(m)
                phi_m_plus_1 = _phi(m + 1)
                
                if phi_m == 0 or phi_m_plus_1 == 0:
                    return float('inf')
                
                return -np.log(phi_m_plus_1 / phi_m)
            
            if len(intervals_hours) >= 20:
                samp_ent = sample_entropy(intervals_hours)
                if not np.isnan(samp_ent) and not np.isinf(samp_ent):
                    st.write(f"**📏 Sample Entropy:** {samp_ent:.3f}")
                    if samp_ent < 0.5:
                        st.success("✅ Alta regularidade - padrão muito consistente")
                    elif samp_ent < 1.0:
                        st.info("📊 Regularidade moderada")
                    else:
                        st.warning("⚠️ Baixa regularidade - padrão irregular")
        
        # ============================================================
        # 12. MATRIZ DE TRANSIÇÃO DE ESTADOS
        # ============================================================
        st.subheader("🔄 12. Análise de Transição de Estados")
        
        if len(intervals_hours) >= 5:
            # Definir estados baseados em quartis
            q1 = np.percentile(intervals_hours, 25)
            q2 = np.percentile(intervals_hours, 50)
            q3 = np.percentile(intervals_hours, 75)
            
            def categorize_interval(interval):
                if interval <= q1:
                    return 'Muito Rápido'
                elif interval <= q2:
                    return 'Rápido'
                elif interval <= q3:
                    return 'Normal'
                else:
                    return 'Lento'
            
            # Categorizar intervalos
            states = [categorize_interval(i) for i in intervals_hours]
            
            # Criar matriz de transição
            state_labels = ['Muito Rápido', 'Rápido', 'Normal', 'Lento']
            transition_matrix = np.zeros((4, 4))
            state_to_idx = {s: i for i, s in enumerate(state_labels)}
            
            for i in range(len(states) - 1):
                current_state = state_to_idx[states[i]]
                next_state = state_to_idx[states[i + 1]]
                transition_matrix[current_state, next_state] += 1
            
            # Normalizar para obter probabilidades
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Evitar divisão por zero
            transition_probs = transition_matrix / row_sums
            
            # Visualizar matriz de transição
            fig_matrix = go.Figure(data=go.Heatmap(
                z=transition_probs,
                x=state_labels,
                y=state_labels,
                text=np.round(transition_probs, 2),
                texttemplate='%{text}',
                colorscale='Blues',
                showscale=True
            ))
            
            fig_matrix.update_layout(
                title="Matriz de Transição de Estados (Probabilidades)",
                xaxis_title="Próximo Estado",
                yaxis_title="Estado Atual",
                height=400
            )
            
            st.plotly_chart(fig_matrix, use_container_width=True, key='transition_matrix')
            
            # Identificar transições mais prováveis
            max_prob_transitions = []
            for i, current in enumerate(state_labels):
                if row_sums[i] > 0:
                    most_likely = state_labels[np.argmax(transition_probs[i])]
                    prob = np.max(transition_probs[i])
                    if prob > 0.4:  # Threshold de 40%
                        max_prob_transitions.append(f"{current} → {most_likely} ({prob:.0%})")
            
            if max_prob_transitions:
                st.success("**🎯 Transições mais prováveis:**")
                for transition in max_prob_transitions:
                    st.write(f"• {transition}")
            
            # Calcular estado estacionário (se existir)
            eigenvalues, eigenvectors = np.linalg.eig(transition_probs.T)
            stationary_idx = np.argmax(np.abs(eigenvalues))
            
            if np.abs(eigenvalues[stationary_idx] - 1.0) < 0.01:
                stationary = np.real(eigenvectors[:, stationary_idx])
                stationary = stationary / stationary.sum()
                
                st.info("**📊 Distribuição de Estado Estacionário (longo prazo):**")
                for i, state in enumerate(state_labels):
                    if stationary[i] > 0.05:  # Mostrar apenas estados relevantes
                        st.write(f"• {state}: {stationary[i]:.1%}")
        
        # ============================================================
        # 13. ANÁLISE DE PONTOS DE MUDANÇA (CHANGE POINTS)
        # ============================================================
        st.subheader("🔀 13. Detecção de Pontos de Mudança")
        
        has_change_points = False
        if len(intervals_hours) >= 20:
            # Usar CUSUM para detectar mudanças
            cumsum = np.cumsum(intervals_hours - np.mean(intervals_hours))
            
            # Detectar pontos de mudança significativos
            threshold = 2 * np.std(intervals_hours) * np.sqrt(len(intervals_hours))
            
            change_points = []
            for i in range(1, len(cumsum) - 1):
                if abs(cumsum[i] - cumsum[i-1]) > threshold/10 or abs(cumsum[i] - cumsum[i+1]) > threshold/10:
                    # Verificar se é um ponto de mudança real
                    before_mean = np.mean(intervals_hours[:i]) if i > 0 else 0
                    after_mean = np.mean(intervals_hours[i:]) if i < len(intervals_hours) else 0
                    
                    if abs(before_mean - after_mean) > np.std(intervals_hours):
                        change_points.append(i)
            
            # Remover pontos muito próximos
            filtered_change_points = []
            for cp in change_points:
                if not filtered_change_points or cp - filtered_change_points[-1] > 5:
                    filtered_change_points.append(cp)
            
            has_change_points = len(filtered_change_points) > 0
            
            if filtered_change_points:
                st.warning(f"⚠️ **{len(filtered_change_points)} pontos de mudança detectados**")
                
                # Visualizar CUSUM com pontos de mudança
                fig_cusum = go.Figure()
                
                fig_cusum.add_trace(go.Scatter(
                    x=list(range(len(cumsum))),
                    y=cumsum,
                    mode='lines',
                    name='CUSUM',
                    line=dict(color='blue', width=2)
                ))
                
                # Adicionar pontos de mudança
                for cp in filtered_change_points:
                    fig_cusum.add_vline(
                        x=cp,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"CP {cp}"
                    )
                
                fig_cusum.update_layout(
                    title="CUSUM com Pontos de Mudança",
                    xaxis_title="Índice",
                    yaxis_title="CUSUM",
                    height=350
                )
                
                st.plotly_chart(fig_cusum, use_container_width=True, key='cusum_plot')
                
                # Analisar períodos entre mudanças
                st.write("**📊 Análise dos Períodos:**")
                periods = [0] + filtered_change_points + [len(intervals_hours)]
                
                for i in range(len(periods) - 1):
                    start, end = periods[i], periods[i+1]
                    period_data = intervals_hours[start:end]
                    if len(period_data) > 0:
                        st.write(f"• **Período {i+1}** ({end-start} alertas): "
                            f"Intervalo médio = {np.mean(period_data):.1f}h ± {np.std(period_data):.1f}h")
                
                # Verificar se há evolução temporal
                period_means = []
                for i in range(len(periods) - 1):
                    start, end = periods[i], periods[i+1]
                    if end > start:
                        period_means.append(np.mean(intervals_hours[start:end]))
                
                if len(period_means) > 1:
                    if all(period_means[i] < period_means[i+1] for i in range(len(period_means)-1)):
                        st.error("📈 **Padrão de degradação:** Intervalos aumentando ao longo do tempo")
                    elif all(period_means[i] > period_means[i+1] for i in range(len(period_means)-1)):
                        st.warning("📉 **Padrão de aceleração:** Intervalos diminuindo ao longo do tempo")
                    else:
                        st.info("🔄 **Padrão variável:** Mudanças não monotônicas")
            else:
                st.success("✅ Sem pontos de mudança significativos - comportamento estável")
        
        # ============================================================
        # 14. CLASSIFICAÇÃO DEFINITIVA: REINCIDENTE vs NÃO REINCIDENTE
        # ============================================================
        st.markdown("---")
        st.header("🎯 14. CLASSIFICAÇÃO FINAL: ALERTA REINCIDENTE?")
        
        # Coletar todas as métricas calculadas
        reincidence_criteria = {}
        reincidence_points = 0
        max_points = 0
        justifications = []
        
        # CRITÉRIO 1: Regularidade dos Intervalos (CV)
        max_points += 20
        if cv < 0.35:
            reincidence_points += 20
            reincidence_criteria['regularidade'] = 'ALTA'
            justifications.append("✅ **Intervalos muito regulares** (CV < 0.35)")
        elif cv < 0.65:
            reincidence_points += 12
            reincidence_criteria['regularidade'] = 'MODERADA'
            justifications.append("🟡 **Intervalos moderadamente regulares** (CV < 0.65)")
        else:
            reincidence_points += 0
            reincidence_criteria['regularidade'] = 'BAIXA'
            justifications.append("❌ **Intervalos irregulares** (CV >= 0.65)")
        
        # CRITÉRIO 2: Score de Recorrência Global
        max_points += 20
        if recurrence_score >= 75:
            reincidence_points += 20
            reincidence_criteria['score_recorrencia'] = 'ALTO'
            justifications.append(f"✅ **Score de recorrência alto** ({recurrence_score}/100)")
        elif recurrence_score >= 50:
            reincidence_points += 12
            reincidence_criteria['score_recorrencia'] = 'MODERADO'
            justifications.append(f"🟡 **Score de recorrência moderado** ({recurrence_score}/100)")
        else:
            reincidence_points += 0
            reincidence_criteria['score_recorrencia'] = 'BAIXO'
            justifications.append(f"❌ **Score de recorrência baixo** ({recurrence_score}/100)")
        
        # CRITÉRIO 3: Periodicidade Detectada (FFT)
        max_points += 15
        if 'dominant_periods' in locals() and len(dominant_periods) > 0:
            reincidence_points += 15
            reincidence_criteria['periodicidade'] = 'SIM'
            justifications.append("✅ **Periodicidade clara detectada** (via FFT)")
        else:
            reincidence_points += 0
            reincidence_criteria['periodicidade'] = 'NÃO'
            justifications.append("❌ **Sem periodicidade detectável**")
        
        # CRITÉRIO 4: Autocorrelação Significativa
        max_points += 15
        if 'significant_peaks' in locals() and significant_peaks:
            reincidence_points += 15
            reincidence_criteria['autocorrelacao'] = 'SIM'
            justifications.append("✅ **Autocorrelação significativa** (padrão repetitivo)")
        else:
            reincidence_points += 0
            reincidence_criteria['autocorrelacao'] = 'NÃO'
            justifications.append("❌ **Sem autocorrelação significativa**")
        
        # CRITÉRIO 5: Concentração Temporal (Hora/Dia)
        max_points += 15
        concentration_detected = False
        
        # Calcular concentração horária se ainda não foi calculada
        if 'total_top_3_hours' not in locals():
            hourly_dist = df_sorted['hour'].value_counts().sort_index()
            if len(hourly_dist) > 0:
                hourly_pct = (hourly_dist / hourly_dist.sum() * 100).round(2)
                top_3_hours = hourly_pct.nlargest(3)
                total_top_3_hours = top_3_hours.sum()
            else:
                total_top_3_hours = 0
        
        # Calcular concentração semanal se ainda não foi calculada
        if 'total_top_3_days' not in locals():
            daily_dist = df_sorted['day_name'].value_counts()
            if len(daily_dist) > 0:
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_dist_ordered = daily_dist.reindex(days_order).fillna(0)
                daily_pct = (daily_dist_ordered / daily_dist_ordered.sum() * 100).round(2)
                top_3_days = daily_pct.nlargest(3)
                total_top_3_days = top_3_days.sum()
            else:
                total_top_3_days = 0
        
        # Avaliar concentração
        if total_top_3_hours > 50:
            concentration_detected = True
            justifications.append(f"✅ **Concentração horária forte** ({total_top_3_hours:.0f}% em top 3 horas)")
        if total_top_3_days > 50:
            concentration_detected = True
            justifications.append(f"✅ **Concentração semanal forte** ({total_top_3_days:.0f}% em top 3 dias)")
        
        if concentration_detected:
            reincidence_points += 15
            reincidence_criteria['concentracao_temporal'] = 'ALTA'
        else:
            reincidence_points += 0
            reincidence_criteria['concentracao_temporal'] = 'BAIXA'
            justifications.append("❌ **Sem concentração temporal clara**")
        
        # CRITÉRIO 6: Entropia (Previsibilidade)
        max_points += 10
        if 'normalized_entropy' in locals():
            if normalized_entropy < 0.5:
                reincidence_points += 10
                reincidence_criteria['previsibilidade'] = 'ALTA'
                justifications.append("✅ **Alta previsibilidade** (baixa entropia)")
            elif normalized_entropy < 0.7:
                reincidence_points += 5
                reincidence_criteria['previsibilidade'] = 'MODERADA'
                justifications.append("🟡 **Previsibilidade moderada**")
            else:
                reincidence_points += 0
                reincidence_criteria['previsibilidade'] = 'BAIXA'
                justifications.append("❌ **Baixa previsibilidade** (alta entropia)")
        
        # CRITÉRIO 7: Ausência de Bursts Irregulares
        max_points += 5
        if 'burst_sequences' in locals():
            if len(burst_sequences) == 0:
                reincidence_points += 5
                reincidence_criteria['bursts'] = 'AUSENTE'
                justifications.append("✅ **Sem padrão de rajadas** (distribuição uniforme)")
            else:
                reincidence_points += 0
                reincidence_criteria['bursts'] = 'PRESENTE'
                justifications.append("❌ **Padrão de rajadas detectado** (comportamento irregular)")
        
        # Calcular percentual final
        reincidence_percentage = (reincidence_points / max_points) * 100 if max_points > 0 else 0
        
        # REGRA DE CLASSIFICAÇÃO FINAL
        st.subheader("📊 Resultado da Análise")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Determinar classificação
            if reincidence_percentage >= 70:
                classification = "🔴 ALERTA REINCIDENTE"
                classification_level = "ALTA"
                color = "red"
                recommendation = """
                **Este alerta apresenta forte padrão de reincidência.**
                
                **Ações Recomendadas:**
                - ✅ Implementar automação de resposta
                - ✅ Criar runbook detalhado
                - ✅ Considerar supressão inteligente em horários previsíveis
                - ✅ Investigar causa raiz para correção definitiva
                - ✅ Monitorar desvios do padrão esperado
                """
            elif reincidence_percentage >= 50:
                classification = "🟠 ALERTA PARCIALMENTE REINCIDENTE"
                classification_level = "MODERADA"
                color = "orange"
                recommendation = """
                **Este alerta apresenta padrão moderado de reincidência.**
                
                **Ações Recomendadas:**
                - 🔍 Investigar causas múltiplas possíveis
                - 📊 Monitorar evolução do padrão
                - ⚙️ Considerar automação parcial
                - 🎯 Focar em períodos de maior concentração
                """
            else:
                classification = "🟢 ALERTA NÃO REINCIDENTE"
                classification_level = "BAIXA"
                color = "green"
                recommendation = """
                **Este alerta NÃO apresenta padrão consistente de reincidência.**
                
                **Ações Recomendadas:**
                - 🔍 Análise caso a caso necessária
                - ❓ Investigar se são falsos positivos
                - 🔧 Revisar configuração do alerta
                - 📉 Considerar desativação se pouco relevante
                - 🎯 Tratar cada ocorrência individualmente
                """
            
            # Mostrar classificação com destaque
            st.markdown(f"### {classification}")
            st.markdown(f"**Nível de Reincidência:** {classification_level}")
            st.markdown(f"**Score:** {reincidence_percentage:.1f}% ({reincidence_points}/{max_points} pontos)")
            
            st.markdown("---")
            st.markdown("#### 📋 Justificativas:")
            for justification in justifications:
                st.markdown(f"- {justification}")
            
            st.markdown("---")
            st.markdown("#### 💡 Recomendações:")
            st.info(recommendation)
        
        with col2:
            # Gauge visual
            fig_final = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = reincidence_percentage,
                title = {'text': "Score de Reincidência", 'font': {'size': 20}},
                delta = {'reference': 50, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "darkred", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_final.update_layout(height=400)
            st.plotly_chart(fig_final, use_container_width=True, key='reincidence_gauge')
            
            # Resumo dos critérios
            st.markdown("#### 📊 Critérios Avaliados:")
            criteria_status = {
                'Regularidade': reincidence_criteria.get('regularidade', 'N/A'),
                'Score Global': reincidence_criteria.get('score_recorrencia', 'N/A'),
                'Periodicidade': reincidence_criteria.get('periodicidade', 'N/A'),
                'Autocorrelação': reincidence_criteria.get('autocorrelacao', 'N/A'),
                'Concentração': reincidence_criteria.get('concentracao_temporal', 'N/A'),
                'Previsibilidade': reincidence_criteria.get('previsibilidade', 'N/A'),
                'Bursts': reincidence_criteria.get('bursts', 'N/A')
            }
            
            for criterion, status in criteria_status.items():
                if status in ['ALTA', 'SIM', 'AUSENTE']:
                    icon = "✅"
                elif status in ['MODERADA', 'MODERADO']:
                    icon = "🟡"
                else:
                    icon = "❌"
                st.markdown(f"{icon} **{criterion}:** {status}")
        
        # Exportar resultado da classificação
        st.markdown("---")
        st.subheader("📥 Exportar Resultado")
        
        result_data = {
            'alert_id': [self.alert_id],
            'classificacao': [classification],
            'nivel_reincidencia': [classification_level],
            'score_percentual': [f"{reincidence_percentage:.1f}%"],
            'pontos': [f"{reincidence_points}/{max_points}"],
            'regularidade_cv': [f"{cv:.3f}"],
            'score_recorrencia': [recurrence_score],
            **{f'criterio_{k}': [v] for k, v in reincidence_criteria.items()}
        }
        
        result_df = pd.DataFrame(result_data)
        
        csv_result = result_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Baixar Classificação (CSV)",
            data=csv_result,
            file_name=f"classificacao_reincidencia_{self.alert_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
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
                                "⏱️ Análise de Intervalos"
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
            3. **Análise de Intervalos:** Regularidade e padrões de tempo
            
            ### Principais Funcionalidades:
            - ✨ Identificação automática de grupos contínuos
            - 📊 Visualização detalhada de grupos com timeline
            - 📈 Análise de recorrência (hora/dia) para alertas contínuos
            - 🎯 Clustering inteligente por perfil de comportamento
            - ⏱️ Detecção de padrões de intervalos (fixo, semi-regular, irregular)
            - 🔴 Separação clara entre alertas isolados e contínuos
            
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
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    # Identificar grupos baseado no gap entre alertas
    alert_data['group_id'] = -1  # -1 = sem grupo (isolado)
    current_group = 0
    group_start_idx = 0
    
    for i in range(len(alert_data)):
        if i == 0:
            continue
            
        gap = alert_data.loc[i, 'time_diff_hours']
        
        # Se o gap for maior que o threshold, finaliza o grupo anterior
        if gap > max_gap_hours:
            # Verifica se o grupo anterior é válido (tamanho mínimo)
            group_size = i - group_start_idx
            if group_size >= min_group_size:
                alert_data.loc[group_start_idx:i-1, 'group_id'] = current_group
                current_group += 1
            # Inicia novo grupo potencial
            group_start_idx = i
    
    # Processa o último grupo
    group_size = len(alert_data) - group_start_idx
    if group_size >= min_group_size:
        alert_data.loc[group_start_idx:, 'group_id'] = current_group
    
    # Detectar spikes desproporcionais dentro de grupos
    alert_data['date'] = alert_data['created_on'].dt.date
    daily_counts = alert_data.groupby('date').size()
    avg_daily = daily_counts.mean()
    spike_threshold = avg_daily * spike_threshold_multiplier
    
    spike_dates = daily_counts[daily_counts > spike_threshold].index
    
    # Marcar alertas em dias de spike como isolados
    if len(spike_dates) > 0:
        alert_data.loc[alert_data['date'].isin(spike_dates), 'group_id'] = -1
    
    # Classificar cada alerta
    alert_data['is_isolated'] = alert_data['group_id'] == -1
    
    # Criar informações dos grupos
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
    Alertas contínuos são aqueles com múltiplos grupos bem definidos.
    Alertas isolados são aqueles sem grupos significativos.
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
            'groups_info': []
        }
    
    # Identificar grupos
    alert_data_processed, groups_info = identify_alert_groups(
        alert_data, max_gap_hours, min_group_size, spike_threshold_multiplier
    )
    
    num_groups = len(groups_info)
    isolated_count = alert_data_processed['is_isolated'].sum()
    grouped_count = n - isolated_count
    
    # Critérios de classificação
    isolated_pct = (isolated_count / n) * 100
    
    # Determinar padrão
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
        'groups_info': groups_info
    }

# ============================================================
# Funções de processamento
# ============================================================
def process_single_alert(alert_id, df_original, max_gap_hours=24, min_group_size=3, 
                        spike_threshold_multiplier=5):
    try:
        df_alert = df_original[df_original['u_alert_id'] == alert_id].copy()
        if len(df_alert) < 1:
            return None
        
        # Classificação de padrão baseada em grupos
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
        df_filtered['month'] = df_filtered['created_on'].dt.month
        df_filtered['month_name'] = df_filtered['created_on'].dt.month_name()
        df_filtered['is_weekend'] = df_filtered['day_of_week'].isin([5, 6])
        df_filtered['is_business_hours'] = (df_filtered['hour'] >= 9) & (df_filtered['hour'] <= 17)
        df_filtered['time_diff_hours'] = df_filtered['created_on'].diff().dt.total_seconds() / 3600
        df_filtered['time_diff_days'] = df_filtered['created_on'].diff().dt.total_seconds() / 86400

        # Identificar grupos de alertas
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
        
        # Estatísticas de padrões
        isolated_count = len(self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'isolated'])
        continuous_count = len(self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'continuous'])
        
        st.subheader("📊 Estatísticas Globais")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
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
        
        return len(self.df_all_alerts) > 0

    def show_isolated_vs_continuous_analysis(self):
        st.header("🔍 Análise de Alertas Isolados vs Contínuos (Baseado em Grupos)")

        # Garantir que não tenha duplicados
        self.df_all_alerts = self.df_all_alerts.drop_duplicates(subset=['alert_id'])

        # Separar alertas
        df_isolated = self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'isolated']
        df_continuous = self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'continuous']

        # Visualização geral
        col1, col2 = st.columns(2)
        with col1:
            # Gráfico de pizza
            pattern_dist = self.df_all_alerts['pattern_type'].value_counts()
            fig_pie = px.pie(
                values=pattern_dist.values,
                names=pattern_dist.index,
                title="📊 Distribuição de Padrões de Alerta",
                color_discrete_map={'isolated': '#ff4444', 'continuous': '#44ff44'}
            )
            st.plotly_chart(fig_pie, use_container_width=True, key='pattern_pie')

        with col2:
            # Estatísticas comparativas
            st.subheader("📈 Comparação de Métricas")
            comparison_data = pd.DataFrame({
                'Métrica': ['Qtd Alertas', 'Média Ocorrências', 'Média Grupos', 
                            'Média % Isolados', 'Média Freq/Dia'],
                'Isolados': [
                    len(df_isolated),
                    df_isolated['total_ocorrencias'].mean() if len(df_isolated) > 0 else 0,
                    df_isolated['num_grupos'].mean() if len(df_isolated) > 0 else 0,
                    df_isolated['pct_isolados'].mean() if len(df_isolated) > 0 else 0,
                    df_isolated['freq_dia'].mean() if len(df_isolated) > 0 else 0
                ],
                'Contínuos': [
                    len(df_continuous),
                    df_continuous['total_ocorrencias'].mean() if len(df_continuous) > 0 else 0,
                    df_continuous['num_grupos'].mean() if len(df_continuous) > 0 else 0,
                    df_continuous['pct_isolados'].mean() if len(df_continuous) > 0 else 0,
                    df_continuous['freq_dia'].mean() if len(df_continuous) > 0 else 0
                ]
            })
            comparison_data = comparison_data.round(2)
            st.dataframe(comparison_data, use_container_width=True)

        # Tabs para detalhes
        tab1, tab2, tab3 = st.tabs(["🔴 Alertas Isolados", "🟢 Alertas Contínuos", "📊 Análise Comparativa"])

        # ISOLADOS
        with tab1:
            st.subheader(f"🔴 Alertas Isolados ({len(df_isolated)} alertas)")

            if len(df_isolated) > 0:
                fig_iso = px.scatter(
                    df_isolated,
                    x='primeiro_alerta',
                    y='total_ocorrencias',
                    size='alertas_isolados',
                    color='pct_isolados',
                    title="⏳ Ocorrências de Alertas Isolados no Tempo",
                    hover_data=['alert_id', 'pattern_reason', 'num_grupos'],
                    labels={'pct_isolados': '% Isolados'}
                )
                st.plotly_chart(fig_iso, use_container_width=True, key='isolated_scatter')

                # Razões para isolamento
                st.write("**📝 Razões para Classificação como Isolado:**")
                reason_counts = df_isolated['pattern_reason'].value_counts()
                for reason, count in reason_counts.items():
                    st.write(f"• {reason}: {count} alertas")

                # Top alertas isolados
                st.write("**🔝 Top 10 Alertas Isolados (por % de alertas isolados):**")
                top_isolated = df_isolated.nlargest(10, 'pct_isolados')[
                    ['alert_id', 'total_ocorrencias', 'alertas_isolados', 'num_grupos', 'pct_isolados', 'pattern_reason']
                ]
                top_isolated.columns = ['Alert ID', 'Total Ocorrências', 'Alertas Isolados', 'Nº Grupos', '% Isolados', 'Razão']
                top_isolated['% Isolados'] = top_isolated['% Isolados'].round(1).astype(str) + '%'
                st.dataframe(top_isolated, use_container_width=True)

                with st.expander("📋 Ver todos os alertas isolados"):
                    isolated_list = df_isolated[['alert_id', 'total_ocorrencias', 'alertas_isolados',
                                                'num_grupos', 'pct_isolados', 'pattern_reason']].copy()
                    isolated_list.columns = ['Alert ID', 'Total', 'Isolados', 'Grupos', '% Isolados', 'Razão']
                    isolated_list['% Isolados'] = isolated_list['% Isolados'].round(1).astype(str) + '%'
                    st.dataframe(isolated_list, use_container_width=True)
            else:
                st.info("Nenhum alerta isolado encontrado com os critérios atuais.")

        # CONTÍNUOS
        with tab2:
            st.subheader(f"🟢 Alertas Contínuos ({len(df_continuous)} alertas)")

            if len(df_continuous) > 0:
                # Top alertas contínuos
                st.write("**🔝 Top 10 Alertas Contínuos (maior número de grupos):**")
                top_continuous = df_continuous.nlargest(10, 'num_grupos')[
                    ['alert_id', 'total_ocorrencias', 'num_grupos', 'alertas_agrupados', 'freq_dia']
                ]
                top_continuous.columns = ['Alert ID', 'Total Ocorrências', 'Nº Grupos', 'Alertas Agrupados', 'Freq/Dia']
                st.dataframe(top_continuous, use_container_width=True)

                # Distribuição de grupos
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
                                                    'alertas_agrupados', 'alertas_isolados', 'pct_isolados']].copy()
                    continuous_list.columns = ['Alert ID', 'Total', 'Grupos', 'Agrupados', 'Isolados', '% Isolados']
                    continuous_list['% Isolados'] = continuous_list['% Isolados'].round(1).astype(str) + '%'
                    st.dataframe(continuous_list, use_container_width=True)
            else:
                st.info("Nenhum alerta contínuo encontrado com os critérios atuais.")

        # ANÁLISE COMPARATIVA
        with tab3:
            st.subheader("📊 Análise Comparativa Detalhada")

            # Scatter plot comparativo
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
                hover_data=['alert_id'],
                color_discrete_map={'isolated': '#ff4444', 'continuous': '#44ff44'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key='comparative_scatter')

            # Box plots comparativos
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

            # Recomendações
            st.subheader("💡 Recomendações de Tratamento")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**🔴 Para Alertas Isolados:**")
                st.write("• Considerar desativação ou revisão de configuração")
                st.write("• Verificar se são falsos positivos")
                st.write("• Analisar contexto específico das ocorrências")
                st.write("• Avaliar consolidação com outros alertas similares")

            with col2:
                st.write("**🟢 Para Alertas Contínuos:**")
                st.write("• Priorizar automação de resposta")
                st.write("• Implementar supressão inteligente")
                st.write("• Criar runbooks específicos")
                st.write("• Considerar ajuste de thresholds")

    def show_global_overview(self, filter_isolated=False):
        st.subheader("📈 Visão Geral dos Alertas")
        
        df_to_analyze = self.df_all_alerts
        if filter_isolated:
            df_to_analyze = self.df_all_alerts[self.df_all_alerts['pattern_type'] == 'continuous']
            st.info(f"🔍 Mostrando apenas alertas contínuos ({len(df_to_analyze)} de {len(self.df_all_alerts)})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🔥 Top 10 Alertas Mais Frequentes**")
            top_frequent = df_to_analyze.nlargest(10, 'total_ocorrencias')[['alert_id', 'total_ocorrencias', 'freq_dia', 'pattern_type']]
            top_frequent.columns = ['Alert ID', 'Total Ocorrências', 'Frequência/Dia', 'Tipo']
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
            fig_int = px.histogram(df_to_analyze, x='freq_mes', title="📊 Distribuição de Frequência (alertas/mes)",
                                  labels={'freq_mes': 'Alertas por mes', 'count': 'Quantidade de Alert IDs'})
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
        inertias = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
            inertias.append(kmeans.inertia_)
        
        optimal_k = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Número Ótimo de Clusters", optimal_k)
        with col2:
            st.metric("📊 Silhouette Score", f"{max(silhouette_scores):.3f}")
        
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans_final.fit_predict(X_scaled)
        
        # Atribuir clusters apenas aos alertas usados no clustering
        df_for_clustering['cluster'] = clusters
        
        # Atualizar o dataframe principal apenas com os índices corretos
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
        cluster_profiles = self.df_all_alerts.groupby('cluster').agg({
            'total_ocorrencias': ['mean', 'std', 'count'],
            'freq_dia': ['mean', 'std'],
            'intervalo_medio_h': ['mean', 'std'],
            'hora_pico': 'mean',
            'pct_fins_semana': 'mean',
            'pct_horario_comercial': 'mean',
            'variabilidade_intervalo': 'mean'
        }).round(2)
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
                variability = cluster_data['variabilidade_intervalo'].mean()
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
                if variability > self.df_all_alerts['variabilidade_intervalo'].median():
                    characteristics.append("📊 **Padrão irregular**")
                else:
                    characteristics.append("📈 **Padrão regular**")
                for char in characteristics:
                    st.write(f"• {char}")
                with st.expander(f"📋 Alertas no Cluster {i}"):
                    cluster_alerts = cluster_data[['alert_id', 'total_ocorrencias', 'freq_dia', 'intervalo_medio_h']].copy()
                    cluster_alerts.columns = ['Alert ID', 'Total Ocorrências', 'Freq/Dia', 'Intervalo Médio (h)']
                    st.dataframe(cluster_alerts, use_container_width=True, key=f'cluster_table_{i}')

    def show_cluster_recommendations(self):
        st.subheader("💡 Recomendações por Cluster")
        for cluster_id in sorted(self.df_all_alerts['cluster'].unique()):
            cluster_data = self.df_all_alerts[self.df_all_alerts['cluster'] == cluster_id]
            avg_freq = cluster_data['freq_dia'].mean()
            avg_interval = cluster_data['intervalo_medio_h'].mean()
            weekend_pct = cluster_data['pct_fins_semana'].mean()
            business_pct = cluster_data['pct_horario_comercial'].mean()
            with st.expander(f"🎯 Recomendações para Cluster {cluster_id} ({len(cluster_data)} alertas)"):
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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔥 Total de Ocorrências", total)
        with col2:
            st.metric("📅 Período (dias)", period_days)
        with col3:
            st.metric("📈 Média/dia", f"{avg_per_day:.2f}")
        with col4:
            last_alert = self.dates.max().strftime("%d/%m %H:%M")
            st.metric("🕐 Último Alerta", last_alert)
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
    
        # Separar alertas isolados e agrupados
        df_isolated = self.df[self.df['is_isolated']]
        df_grouped = self.df[~self.df['is_isolated']]
    
        # Estatísticas gerais
        st.subheader("📊 Estatísticas Gerais do Alert ID")
        col1, col2, col3, col4, col5 = st.columns(5)
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
    
        # Informações dos grupos
        if len(self.groups_info) > 0:
            st.subheader("📦 Informações dos Grupos")
            groups_df = pd.DataFrame(self.groups_info)
            groups_df['start_time'] = pd.to_datetime(groups_df['start_time']).dt.strftime('%Y-%m-%d %H:%M')
            groups_df['end_time'] = pd.to_datetime(groups_df['end_time']).dt.strftime('%Y-%m-%d %H:%M')
            groups_df['duration_hours'] = groups_df['duration_hours'].round(2)
            groups_df.columns = ['ID Grupo', 'Tamanho', 'Início', 'Fim', 'Duração (h)']
            st.dataframe(groups_df, use_container_width=True)
    
        tab1, tab2, tab3 = st.tabs(["🔴 Ocorrências Isoladas", "🟢 Ocorrências Agrupadas", "📊 Visualização Temporal"])
    
        with tab1:
            st.subheader(f"🔴 Ocorrências Isoladas ({len(df_isolated)})")
            if len(df_isolated) > 0:
                isolated_display = df_isolated[['created_on', 'hour', 'day_name', 'time_diff_hours', 'date']].copy()
                isolated_display['created_on'] = isolated_display['created_on'].dt.strftime('%Y-%m-%d %H:%M:%S')
                isolated_display.columns = ['Data/Hora', 'Hora', 'Dia da Semana', 'Intervalo (h)', 'Data']
                st.dataframe(isolated_display, use_container_width=True)
                st.write(f"**Percentual:** {len(df_isolated)/len(self.df)*100:.2f}% das ocorrências são isoladas")
                
                # Detectar spikes (dias com muitos alertas)
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
            
            # Gráfico de timeline com grupos coloridos
            fig = go.Figure()
            
            # Adicionar alertas isolados
            if len(df_isolated) > 0:
                fig.add_trace(go.Scatter(
                    x=df_isolated['created_on'],
                    y=[1] * len(df_isolated),
                    mode='markers',
                    name='Isolados',
                    marker=dict(size=10, color='red', symbol='x'),
                    hovertemplate='%{x}<br>Isolado<extra></extra>'
                ))
            
            # Adicionar alertas agrupados por grupo
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
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig, use_container_width=True)

    def show_temporal_patterns(self):
        st.header("⏰ Padrões Temporais")
        col1, col2 = st.columns(2)
        with col1:
            hourly = self.df['hour'].value_counts().sort_index()
            fig_hour = px.bar(
                x=hourly.index, 
                y=hourly.values,
                title="📊 Distribuição por Hora do Dia",
                labels={'x': 'Hora', 'y': 'Quantidade de Alertas'}
            )
            fig_hour.update_layout(showlegend=False)
            st.plotly_chart(fig_hour, use_container_width=True, key='hourly_dist')
            peak_hour = hourly.idxmax()
            quiet_hour = hourly.idxmin()
            st.write(f"🕐 **Pico:** {peak_hour:02d}:00 ({hourly[peak_hour]} alertas)")
            st.write(f"🌙 **Menor atividade:** {quiet_hour:02d}:00 ({hourly[quiet_hour]} alertas)")
        with col2:
            daily = self.df['day_name'].value_counts()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_ordered = daily.reindex(days_order).fillna(0)
            fig_day = px.bar(
                x=daily_ordered.index, 
                y=daily_ordered.values,
                title="📅 Distribuição por Dia da Semana",
                labels={'x': 'Dia', 'y': 'Quantidade de Alertas'}
            )
            fig_day.update_layout(showlegend=False)
            st.plotly_chart(fig_day, use_container_width=True, key='daily_dist')
            busiest_day = daily.idxmax()
            st.write(f"📈 **Dia mais ativo:** {busiest_day} ({daily[busiest_day]} alertas)")
        col1, col2 = st.columns(2)
        with col1:
            business = self.df['is_business_hours'].sum()
            non_business = len(self.df) - business
            st.subheader("🏢 Horário Comercial (9h-17h)")
            business_data = pd.DataFrame({
                'Período': ['Comercial', 'Fora do horário'],
                'Quantidade': [business, non_business],
                'Porcentagem': [business/len(self.df)*100, non_business/len(self.df)*100]
            })
            fig_business = px.pie(
                business_data, 
                values='Quantidade', 
                names='Período',
                title="Distribuição por Horário"
            )
            st.plotly_chart(fig_business, use_container_width=True, key='business_hours_pie')
        with col2:
            weekend = self.df['is_weekend'].sum()
            weekday = len(self.df) - weekend
            st.subheader("🗓️ Fins de Semana vs Dias Úteis")
            weekend_data = pd.DataFrame({
                'Período': ['Dias úteis', 'Fins de semana'],
                'Quantidade': [weekday, weekend],
                'Porcentagem': [weekday/len(self.df)*100, weekend/len(self.df)*100]
            })
            fig_weekend = px.pie(
                weekend_data, 
                values='Quantidade', 
                names='Período',
                title="Distribuição Semanal"
            )
            st.plotly_chart(fig_weekend, use_container_width=True, key='weekend_pie')

    def show_burst_analysis(self):
        st.header("💥 Análise de Rajadas")
        burst_threshold = st.slider("⏱️ Threshold para Rajada (horas)", 0.5, 24.0, 2.0, 0.5)
        intervals = self.df[~self.df['is_isolated']]['time_diff_hours'].fillna(999)
        bursts, current_burst = [], []
        for i, interval in enumerate(intervals):
            if interval <= burst_threshold and i > 0:
                if not current_burst:
                    current_burst = [i-1, i]
                else:
                    current_burst.append(i)
            else:
                if len(current_burst) >= 2:
                    bursts.append(current_burst)
                current_burst = []
        if len(current_burst) >= 2:
            bursts.append(current_burst)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚨 Rajadas Detectadas", len(bursts))
        if bursts:
            burst_sizes = [len(b) for b in bursts]
            with col2:
                st.metric("📊 Tamanho Médio", f"{np.mean(burst_sizes):.1f}")
            with col3:
                st.metric("📈 Maior Rajada", f"{max(burst_sizes)} alertas")
            st.subheader("🔥 Maiores Rajadas")
            sorted_bursts = sorted(bursts, key=len, reverse=True)[:5]
            burst_data = []
            for i, burst_indices in enumerate(sorted_bursts):
                start_time = self.df.iloc[burst_indices[0]]['created_on']
                end_time = self.df.iloc[burst_indices[-1]]['created_on']
                duration = end_time - start_time
                burst_data.append({
                    'Rajada': f"#{i+1}",
                    'Alertas': len(burst_indices),
                    'Início': start_time.strftime("%d/%m/%Y %H:%M"),
                    'Fim': end_time.strftime("%d/%m/%Y %H:%M"),
                    'Duração': str(duration)
                })
            st.dataframe(pd.DataFrame(burst_data), use_container_width=True)

    def show_trend_analysis(self):
        st.header("📈 Análise de Tendências")
        daily_counts = self.df.groupby('date').size()
        if len(daily_counts) >= 7:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_counts.index,
                y=daily_counts.values,
                mode='lines+markers',
                name='Alertas por dia',
                line=dict(color='blue')
            ))
            x_numeric = np.arange(len(daily_counts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, daily_counts.values)
            trend_line = slope * x_numeric + intercept
            fig.add_trace(go.Scatter(
                x=daily_counts.index,
                y=trend_line,
                mode='lines',
                name='Tendência',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title="📊 Evolução Temporal dos Alertas",
                xaxis_title="Data",
                yaxis_title="Número de Alertas",
                hovermode='x'
            )
            st.plotly_chart(fig, use_container_width=True, key='trend_analysis')
            if slope > 0.01:
                trend = "CRESCENTE 📈"
            elif slope < -0.01:
                trend = "DECRESCENTE 📉"
            else:
                trend = "ESTÁVEL ➡️"
            strength = "FORTE" if abs(r_value) > 0.7 else "MODERADA" if abs(r_value) > 0.3 else "FRACA"
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Tendência", trend)
            with col2:
                st.metric("💪 Força", strength)
            with col3:
                st.metric("📊 Correlação", f"{r_value:.4f}")
            with col4:
                st.metric("⚡ Taxa/dia", f"{slope:.4f}")
        else:
            st.warning("⚠️ Poucos dados para análise de tendência (mínimo 7 dias)")

    def show_anomaly_detection(self):
        st.header("🚨 Detecção de Anomalias")
        intervals = self.df['time_diff_hours'].dropna()
        if len(intervals) > 4:
            Q1 = intervals.quantile(0.25)
            Q3 = intervals.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            fast_anomalies = intervals[intervals < lower_bound]
            slow_anomalies = intervals[intervals > upper_bound]
            normal_intervals = intervals[(intervals >= lower_bound) & (intervals <= upper_bound)]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⚡ Intervalos Curtos", len(fast_anomalies))
            with col2:
                st.metric("🐌 Intervalos Longos", len(slow_anomalies))
            with col3:
                st.metric("✅ Intervalos Normais", len(normal_intervals))
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=intervals,
                name="Intervalos (horas)",
                boxpoints='outliers'
            ))
            fig.update_layout(
                title="📊 Distribuição dos Intervalos (Detecção de Outliers)",
                yaxis_title="Horas"
            )
            st.plotly_chart(fig, use_container_width=True, key='anomaly_boxplot')
            if len(fast_anomalies) > 0 or len(slow_anomalies) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    if len(fast_anomalies) > 0:
                        st.subheader("⚡ Intervalos Muito Curtos")
                        st.write(f"Menor intervalo: **{fast_anomalies.min():.2f} horas**")
                        st.write(f"Média dos curtos: **{fast_anomalies.mean():.2f} horas**")
                with col2:
                    if len(slow_anomalies) > 0:
                        st.subheader("🐌 Intervalos Muito Longos")
                        st.write(f"Maior intervalo: **{slow_anomalies.max():.2f} horas**")
                        st.write(f"Média dos longos: **{slow_anomalies.mean():.2f} horas**")
        else:
            st.warning("⚠️ Poucos dados para detecção de anomalias")

    def show_predictions(self):
        st.header("🔮 Insights Preditivos")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⏰ Horários de Maior Probabilidade")
            hourly_prob = self.df['hour'].value_counts(normalize=True).sort_values(ascending=False)
            prob_data = [{'Horário': f"{hour:02d}:00", 'Probabilidade': f"{prob*100:.1f}%"} for hour, prob in hourly_prob.head(5).items()]
            st.dataframe(pd.DataFrame(prob_data), use_container_width=True)
        with col2:
            st.subheader("📅 Dias de Maior Probabilidade")
            daily_prob = self.df['day_name'].value_counts(normalize=True).sort_values(ascending=False)
            day_data = [{'Dia': day, 'Probabilidade': f"{prob*100:.1f}%"} for day, prob in daily_prob.items()]
            st.dataframe(pd.DataFrame(day_data), use_container_width=True)
        st.subheader("⏱️ Previsão do Próximo Alerta")
        intervals = self.df['time_diff_hours'].dropna()
        if len(intervals) > 0:
            avg_interval = intervals.mean()
            median_interval = intervals.median()
            last_alert = self.dates.max()
            next_avg = last_alert + timedelta(hours=avg_interval)
            next_median = last_alert + timedelta(hours=median_interval)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🕐 Último Alerta", last_alert.strftime("%d/%m %H:%M"))
            with col2:
                st.metric("📊 Próximo (Média)", next_avg.strftime("%d/%m %H:%M"))
            with col3:
                st.metric("📈 Próximo (Mediana)", next_median.strftime("%d/%m %H:%M"))
            st.info(f"💡 **Baseado em:** Intervalo médio de {avg_interval:.1f}h e mediana de {median_interval:.1f}h")

def main():
    st.title("🚨 Analisador de Alertas - Versão Completa")
    st.markdown("### Análise individual, global e agrupamento inteligente de alertas")
    st.sidebar.header("⚙️ Configurações")
    
    # Parâmetros de agrupamento
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
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "📊 Visão Geral",
                            "🔍 Isolados vs Contínuos",
                            "🎯 Agrupamento", 
                            "👥 Perfis dos Clusters",
                            "💡 Recomendações"
                        ])
                        with tab1:
                            analyzer.show_global_overview()
                        with tab2:
                            analyzer.show_isolated_vs_continuous_analysis()
                        with tab3:
                            n_clusters = analyzer.perform_clustering_analysis()
                        with tab4:
                            if n_clusters:
                                analyzer.show_cluster_profiles(n_clusters)
                        with tab5:
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
                    unique_ids = analyzer.df_original['u_alert_id'].unique()
                    selected_id = st.sidebar.selectbox(
                        "🎯 Selecione o Alert ID",
                        unique_ids,
                        help="Escolha o ID do alerta para análise"
                    )
                    if st.sidebar.button("🚀 Executar Análise Individual", type="primary"):
                        analyzer.max_gap_hours = max_gap_hours
                        analyzer.min_group_size = min_group_size
                        analyzer.spike_threshold_multiplier = spike_threshold_multiplier
                        
                        if analyzer.prepare_individual_analysis(selected_id):
                            st.success(f"🎯 Analisando alert_id: {selected_id} ({len(analyzer.df)} registros)")
                            st.info(f"📅 **Período analisado:** {analyzer.dates.min()} até {analyzer.dates.max()}")
                            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                                "🔍 Isolados vs Agrupados",
                                "📊 Básico", 
                                "⏰ Temporais", 
                                "💥 Rajadas", 
                                "📈 Tendências", 
                                "🚨 Anomalias", 
                                "🔮 Previsões"
                            ])
                            with tab1:
                                analyzer.show_individual_alert_analysis()
                            with tab2:
                                analyzer.show_basic_stats()
                            with tab3:
                                analyzer.show_temporal_patterns()
                            with tab4:
                                analyzer.show_burst_analysis()
                            with tab5:
                                analyzer.show_trend_analysis()
                            with tab6:
                                analyzer.show_anomaly_detection()
                            with tab7:
                                analyzer.show_predictions()
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
            
            #### 🎛️ **Parâmetros de Agrupamento** (Configurável na Sidebar)
            
            **⏱️ Gap Máximo Entre Alertas:**
            - Define o intervalo máximo (em horas) entre alertas para considerá-los do mesmo grupo
            - Padrão: 24 horas
            - Alertas separados por mais tempo são considerados de grupos diferentes
            
            **📊 Tamanho Mínimo do Grupo:**
            - Número mínimo de alertas necessários para formar um grupo válido
            - Padrão: 3 alertas
            - Grupos menores são desconsiderados e seus alertas marcados como isolados
            
            **🚀 Multiplicador de Spike:**
            - Define quando um dia tem alertas "desproporcionais"
            - Padrão: 5x a média diária
            - Dias com alertas acima deste threshold são marcados como spikes isolados
            
            #### 🌍 **Análise Global**
            1. **📁 Upload do arquivo:** Carregue um arquivo CSV com os dados dos alertas
            2. **🎯 Selecione "Análise Global"** no modo de análise
            3. **🎛️ Ajuste os parâmetros** de agrupamento se necessário
            4. **⚡ Ative Multiprocessing** para processamento mais rápido (recomendado)
            5. **🚀 Clique em "Executar Análise Global"**
            6. **📊 Explore os resultados** nas diferentes abas:
               - **Visão Geral:** Estatísticas gerais e top alertas
               - **Isolados vs Contínuos:** Comparação detalhada de padrões baseados em grupos
               - **Agrupamento:** Clustering por comportamento
               - **Perfis dos Clusters:** Características de cada grupo
               - **Recomendações:** Sugestões de ação
            
            #### 🔍 **Análise Individual**
            1. **📁 Upload do arquivo:** Carregue um arquivo CSV com os dados dos alertas
            2. **🎯 Selecione "Análise Individual"** no modo de análise
            3. **🎛️ Ajuste os parâmetros** de agrupamento se necessário
            4. **🎯 Escolha um Alert ID** específico
            5. **🚀 Clique em "Executar Análise Individual"**
            6. **📊 Navegue pelas abas** para ver diferentes análises:
               - **Isolados vs Agrupados:** Visualização dos grupos identificados
               - **Básico:** Estatísticas fundamentais
               - **Temporais:** Padrões de horário e dia
               - **Rajadas:** Detecção de bursts
               - **Tendências:** Evolução temporal
               - **Anomalias:** Outliers detectados
               - **Previsões:** Insights preditivos
            
            ### Colunas necessárias no CSV:
            - `u_alert_id`: Identificador único do alerta
            - `created_on`: Data e hora da criação do alerta
            
            ### 🆕 **Nova Lógica de Classificação:**
            
            **📦 Baseada em Grupos:**
            - Alertas são agrupados baseado no gap de tempo entre eles
            - Grupos pequenos (< tamanho mínimo) são desconsiderados
            - Dias com spikes desproporcionais são isolados automaticamente
            
            **🔴 Alertas Isolados:**
            - Sem grupos válidos identificados
            - Maioria das ocorrências são esparsas
            - Incluem dias com explosão anormal de alertas
            
            **🟢 Alertas Contínuos:**
            - 2+ grupos bem definidos identificados
            - Padrão recorrente e consistente
            - Maioria das ocorrências estão agrupadas
            
            ### 🚀 **Multiprocessing!**
            - **⚡ Processamento Paralelo:** Usa múltiplos núcleos da CPU
            - **📈 Muito mais rápido:** Ideal para grandes volumes de dados
            - **🔧 Automático:** Detecta número ideal de processos
            - **💾 Fallback seguro:** Volta para modo sequencial se houver problemas
            """)

if __name__ == "__main__":
    main()
class AdvancedRecurrenceAnalyzer:
    def __init__(self, df, alert_id):
        self.df = df.copy() if df is not None else None
        self.alert_id = alert_id
        self.cluster_threshold_minutes = 5  # 5 minutos para agrupar clusters

    def _detect_and_group_clusters(self, df):
        """
        Detecta clusters de alertas (ocorrências muito próximas no tempo)
        e agrupa como um único evento para análise mais precisa.
        """
        if df is None or len(df) == 0:
            return df, 0, 1.0
        
        df = df.sort_values('created_on').copy()
        df['time_diff_minutes'] = df['created_on'].diff().dt.total_seconds() / 60
        
        # Identificar clusters (diferença <= 5 minutos)
        df['is_cluster_start'] = (df['time_diff_minutes'].isna()) | (df['time_diff_minutes'] > self.cluster_threshold_minutes)
        df['event_id'] = df['is_cluster_start'].cumsum()
        
        # Estatísticas de clusters
        total_occurrences = len(df)
        total_events = df['event_id'].nunique()
        cluster_ratio = total_occurrences / total_events if total_events > 0 else 1.0
        
        # Agregar clusters em eventos únicos
        events_df = df.groupby('event_id').agg({
            'created_on': 'first',  # Usa o primeiro timestamp do cluster
            'u_alert_id': 'first',
            'priority': lambda x: list(x.unique()),
            'clear': 'sum' if 'clear' in df.columns else 'count',
            'qtde_atuacao_manual_maior_0': 'sum' if 'qtde_atuacao_manual_maior_0' in df.columns else 'count'
        }).reset_index(drop=True)
        
        events_df['cluster_size'] = df.groupby('event_id').size().values
        
        return events_df, total_events, cluster_ratio

    def _prepare_data(self):
        """Prepara dados identificando clusters e calculando métricas."""
        if self.df is None or len(self.df) < 3:
            return None, None, None
        
        df = self.df.sort_values('created_on').copy()
        df['created_on'] = pd.to_datetime(df['created_on'], errors='coerce')
        df = df.dropna(subset=['created_on'])
        
        # Detectar e agrupar clusters
        events_df, total_events, cluster_ratio = self._detect_and_group_clusters(df)
        
        if events_df is None or len(events_df) < 2:
            return None, None, None
        
        # Calcular métricas baseadas em EVENTOS (não ocorrências brutas)
        events_df['timestamp'] = events_df['created_on'].astype('int64') // 10**9
        events_df['time_diff_seconds'] = events_df['timestamp'].diff()
        events_df['time_diff_hours'] = events_df['time_diff_seconds'] / 3600
        
        dt = events_df['created_on'].dt
        events_df['hour'] = dt.hour
        events_df['day_of_week'] = dt.dayofweek
        events_df['day_of_month'] = dt.day
        events_df['week_of_year'] = dt.isocalendar().week
        events_df['month'] = dt.month
        events_df['day_name'] = dt.day_name()
        events_df['is_weekend'] = events_df['day_of_week'].isin([5, 6])
        events_df['is_business_hours'] = (events_df['hour'] >= 9) & (events_df['hour'] <= 17)
        
        return events_df, total_events, cluster_ratio

    def analyze(self):
        """Modo interativo (Streamlit) com detecção de clusters."""
        st.header("🔄 Análise Avançada de Reincidência Temporal")
        
        df_events, total_events, cluster_ratio = self._prepare_data()
        
        if df_events is None:
            st.warning("⚠️ Dados insuficientes (mínimo 3 ocorrências).")
            return

        total_occurrences = len(self.df)
        
        st.info(f"📊 Analisando Short CI: **{self.alert_id}**")
        
        # Mostrar info de clusters
        col1, col2, col3 = st.columns(3)
        col1.metric("🔥 Total de Ocorrências", total_occurrences)
        col2.metric("📦 Eventos (após agrupar clusters)", total_events)
        col3.metric("📊 Ratio Cluster", f"{cluster_ratio:.2f}x")
        
        if cluster_ratio > 2.0:
            st.warning(f"⚠️ **Alta taxa de clusters detectada ({cluster_ratio:.1f}x)** - Indica problema gerando múltiplos alertas simultâneos, não reincidência temporal!")
        elif cluster_ratio > 1.5:
            st.info(f"📊 Clusters moderados detectados ({cluster_ratio:.1f}x)")
        else:
            st.success(f"✅ Baixa taxa de clusters ({cluster_ratio:.1f}x) - Boa distribuição temporal")

        if 'priority' in self.df.columns:
            unique_priorities = self.df['priority'].dropna().unique()
            if len(unique_priorities) > 0:
                priorities_str = ', '.join(sorted([str(p) for p in unique_priorities]))
                st.info(f"🎯 **Prioridades:** {priorities_str}")

        if 'clear' in self.df.columns:
            total_clears = int(self.df['clear'].sum())
            clear_percentage = (total_clears / total_occurrences * 100) if total_occurrences > 0 else 0
            st.markdown("---")
            st.subheader("🔒 Análise de Encerramento (Clear)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Clears", total_clears)
            col2.metric("% Encerrado por Clear", f"{clear_percentage:.1f}%")
            col3.metric("% Sem Clear", f"{100 - clear_percentage:.1f}%")

        intervals_hours = df_events['time_diff_hours'].dropna().values
        if len(intervals_hours) < 2:
            st.warning("⚠️ Intervalos insuficientes entre eventos.")
            return

        results = {}
        results['basic_stats'] = self._analyze_basic_statistics(intervals_hours, render=True)
        results['regularity'] = self._analyze_regularity(intervals_hours, render=True)
        results['periodicity'] = self._analyze_periodicity(intervals_hours, render=True)
        results['temporal'] = self._analyze_temporal_patterns(df_events, render=True)
        results['predictability'] = self._calculate_predictability(intervals_hours, render=True)

        self._final_classification(results, df_events, intervals_hours, total_events, cluster_ratio)

    def analyze_complete_silent(self):
        """Modo silencioso para processamento em lote."""
        df_events, total_events, cluster_ratio = self._prepare_data()

        priorities_list = []
        if self.df is not None and 'priority' in self.df.columns:
            unique_priorities = self.df['priority'].dropna().unique().tolist()
            priorities_list = sorted([str(p) for p in unique_priorities])

        total_occurrences = len(self.df) if self.df is not None else 0
        
        total_atuacao_manual = 0
        atuacao_manual_percentage = 0.0
        if self.df is not None and 'qtde_atuacao_manual_maior_0' in self.df.columns:
            total_atuacao_manual = int(self.df['qtde_atuacao_manual_maior_0'].sum())
            registros_com_atuacao = (self.df['qtde_atuacao_manual_maior_0'] > 0).sum()
            atuacao_manual_percentage = float((registros_com_atuacao / total_occurrences * 100) if total_occurrences > 0 else 0)

        if df_events is None or len(df_events) < 3 or total_events < 3:
            total_clears = 0
            clear_percentage = 0.0
            if self.df is not None and 'clear' in self.df.columns:
                total_clears = int(self.df['clear'].sum())
                clear_percentage = float((total_clears / total_occurrences * 100) if total_occurrences > 0 else 0)
            
            return {
                'u_alert_id': self.alert_id,
                'total_occurrences': total_occurrences,
                'total_events': total_events if total_events else 0,
                'cluster_ratio': cluster_ratio if cluster_ratio else 1.0,
                'score': 0,
                'classification': '⚪ DADOS INSUFICIENTES (R4)',
                'mean_interval_hours': None,
                'median_interval_hours': None,
                'cv': None,
                'regularity_score': 0,
                'periodicity_detected': False,
                'dominant_period_hours': None,
                'predictability_score': 0,
                'next_occurrence_prediction_hours': None,
                'hourly_concentration': 0,
                'daily_concentration': 0,
                'total_clears': total_clears,
                'clear_percentage': clear_percentage,
                'total_atuacao_manual': total_atuacao_manual,
                'atuacao_manual_percentage': atuacao_manual_percentage,
                'priorities': priorities_list
            }

        intervals_hours = df_events['time_diff_hours'].dropna().values

        if len(intervals_hours) < 2:
            total_clears = 0
            clear_percentage = 0.0
            if self.df is not None and 'clear' in self.df.columns:
                total_clears = int(self.df['clear'].sum())
                clear_percentage = float((total_clears / total_occurrences * 100) if total_occurrences > 0 else 0)
            
            return {
                'u_alert_id': self.alert_id,
                'total_occurrences': total_occurrences,
                'total_events': total_events,
                'cluster_ratio': cluster_ratio,
                'score': 0,
                'classification': '⚪ INTERVALOS INSUFICIENTES (R4)',
                'mean_interval_hours': None,
                'median_interval_hours': None,
                'cv': None,
                'regularity_score': 0,
                'periodicity_detected': False,
                'dominant_period_hours': None,
                'predictability_score': 0,
                'next_occurrence_prediction_hours': None,
                'hourly_concentration': 0,
                'daily_concentration': 0,
                'total_clears': total_clears,
                'clear_percentage': clear_percentage,
                'total_atuacao_manual': total_atuacao_manual,
                'atuacao_manual_percentage': atuacao_manual_percentage,
                'priorities': priorities_list
            }

        results = {}
        try:
            results['basic_stats'] = self._analyze_basic_statistics(intervals_hours, render=False)
        except Exception:
            results['basic_stats'] = {'mean': 0, 'median': 0, 'std': 0, 'cv': 0}

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
            results['temporal'] = self._analyze_temporal_patterns(df_events, render=False)
        except Exception:
            results['temporal'] = {'hourly_concentration': 0, 'daily_concentration': 0, 'peak_hours': [], 'peak_days': []}

        total_clears = 0
        clear_percentage = 0.0
        if self.df is not None and 'clear' in self.df.columns:
            total_clears = int(self.df['clear'].sum())
            clear_percentage = float((total_clears / total_occurrences * 100) if total_occurrences > 0 else 0)

        final_score, classification = self._calculate_final_score_validated(
            results, df_events, intervals_hours, total_events, cluster_ratio
        )

        return {
            'u_alert_id': self.alert_id,
            'total_occurrences': total_occurrences,
            'total_events': total_events,
            'cluster_ratio': round(cluster_ratio, 2),
            'score': final_score,
            'classification': classification,
            'mean_interval_hours': results['basic_stats'].get('mean'),
            'median_interval_hours': results['basic_stats'].get('median'),
            'cv': results['basic_stats'].get('cv'),
            'regularity_score': results['regularity'].get('regularity_score'),
            'periodicity_detected': results['periodicity'].get('has_strong_periodicity', False),
            'dominant_period_hours': results['periodicity'].get('dominant_period_hours'),
            'predictability_score': results['predictability'].get('predictability_score'),
            'next_occurrence_prediction_hours': results['predictability'].get('next_expected_hours'),
            'hourly_concentration': results['temporal'].get('hourly_concentration'),
            'daily_concentration': results['temporal'].get('daily_concentration'),
            'total_clears': total_clears,
            'clear_percentage': clear_percentage,
            'total_atuacao_manual': total_atuacao_manual,
            'atuacao_manual_percentage': atuacao_manual_percentage,
            'priorities': priorities_list
        }

    def _calculate_final_score_validated(self, results, df_events, intervals_hours, total_events, cluster_ratio):
        """
        Calcula score com critérios RIGOROSOS considerando detecção de clusters.
        
        R1 = REINCIDÊNCIA CRÍTICA (padrão forte e previsível)
        R2 = REINCIDÊNCIA PARCIAL (padrão moderado)
        R3 = REINCIDÊNCIA DETECTÁVEL (sinais fracos)
        R4 = NÃO REINCIDENTE (sem padrão ou dominado por clusters)
        """
        score = 0
        reasons = []
        
        # ====== CRITÉRIO ELIMINATÓRIO: CLUSTERS ======
        # Se ratio de cluster é muito alto, provavelmente não é reincidência temporal
        if cluster_ratio >= 3.0:
            reasons.append(f"❌ ELIMINADO: Alta taxa de clusters ({cluster_ratio:.1f}x) - Problema gerando múltiplos alertas simultâneos")
            return 0, f"⚪ NÃO REINCIDENTE (R4) - Clusters dominantes"
        
        # ====== 1. VOLUME DE EVENTOS (não ocorrências brutas) ======
        if total_events >= 20:
            score += 25
            reasons.append(f"✅ Volume ALTO de eventos ({total_events})")
        elif total_events >= 15:
            score += 20
            reasons.append(f"✅ Volume BOM de eventos ({total_events})")
        elif total_events >= 10:
            score += 12
            reasons.append(f"⚠️ Volume MÉDIO de eventos ({total_events})")
        elif total_events >= 5:
            score += 5
            reasons.append(f"⚠️ Volume BAIXO de eventos ({total_events})")
        else:
            reasons.append(f"❌ Volume insuficiente ({total_events} eventos)")
            return 0, f"⚪ NÃO REINCIDENTE (R4) - Volume insuficiente"
        
        # Penalizar se cluster ratio for alto (mas não eliminatório)
        if cluster_ratio >= 2.0:
            score -= 10
            reasons.append(f"⚠️ PENALIDADE: Clusters moderados ({cluster_ratio:.1f}x)")
        
        # ====== 2. REGULARIDADE (CV nos intervalos entre eventos) ======
        cv = results['regularity'].get('cv', float('inf'))
        
        if cv < 0.25:  # Muito regular
            score += 30
            reasons.append(f"✅ MUITO REGULAR (CV={cv:.3f})")
        elif cv < 0.40:  # Regular
            score += 22
            reasons.append(f"✅ REGULAR (CV={cv:.3f})")
        elif cv < 0.60:  # Semi-regular
            score += 12
            reasons.append(f"⚠️ SEMI-REGULAR (CV={cv:.3f})")
        elif cv < 1.0:  # Irregular
            score += 4
            reasons.append(f"⚠️ IRREGULAR (CV={cv:.3f})")
        else:  # Muito irregular
            reasons.append(f"❌ MUITO IRREGULAR (CV={cv:.3f})")
        
        # ====== 3. PERIODICIDADE (FFT) ======
        has_strong_periodicity = results['periodicity'].get('has_strong_periodicity', False)
        has_moderate_periodicity = results['periodicity'].get('has_moderate_periodicity', False)
        dominant_period = results['periodicity'].get('dominant_period_hours')
        
        if has_strong_periodicity and dominant_period:
            score += 25
            period_str = f"{dominant_period:.1f}h" if dominant_period < 24 else f"{dominant_period/24:.1f}d"
            reasons.append(f"✅ PERIODICIDADE FORTE (~{period_str})")
        elif has_moderate_periodicity:
            score += 12
            reasons.append(f"⚠️ PERIODICIDADE MODERADA")
        else:
            reasons.append(f"❌ SEM PERIODICIDADE")
        
        # ====== 4. CONCENTRAÇÃO TEMPORAL ======
        hourly_conc = results['temporal'].get('hourly_concentration', 0)
        daily_conc = results['temporal'].get('daily_concentration', 0)
        
        if hourly_conc > 65:
            score += 15
            reasons.append(f"✅ Alta concentração horária ({hourly_conc:.1f}%)")
        elif hourly_conc > 45:
            score += 8
            reasons.append(f"⚠️ Média concentração horária ({hourly_conc:.1f}%)")
        
        if daily_conc > 65:
            score += 12
            reasons.append(f"✅ Alta concentração em dias ({daily_conc:.1f}%)")
        elif daily_conc > 45:
            score += 6
            reasons.append(f"⚠️ Média concentração em dias ({daily_conc:.1f}%)")
        
        # ====== 5. PREVISIBILIDADE ======
        predictability = results['predictability'].get('predictability_score', 0)
        
        if predictability > 75:
            score += 20
            reasons.append(f"✅ ALTA previsibilidade ({predictability:.0f}%)")
        elif predictability > 55:
            score += 12
            reasons.append(f"✅ BOA previsibilidade ({predictability:.0f}%)")
        elif predictability > 35:
            score += 5
            reasons.append(f"⚠️ Previsibilidade MÉDIA ({predictability:.0f}%)")
        else:
            reasons.append(f"❌ BAIXA previsibilidade ({predictability:.0f}%)")
        
        # ====== 6. INTERVALO MÉDIO ======
        mean_interval = results['basic_stats'].get('mean')
        if mean_interval:
            if mean_interval < 24:  # < 1 dia
                score += 8
                reasons.append(f"✅ Intervalos curtos ({mean_interval:.1f}h)")
            elif mean_interval < 72:  # < 3 dias
                score += 4
                reasons.append(f"⚠️ Intervalos médios ({mean_interval:.1f}h)")
            elif mean_interval > 168:  # > 1 semana
                score -= 8
                reasons.append(f"⚠️ Intervalos LONGOS ({mean_interval/24:.1f}d)")
        
        # ====== CLASSIFICAÇÃO FINAL ======
        # Score máximo possível: ~137 pontos (sem penalidades)
        # Thresholds ajustados para serem mais rigorosos
        
        # R1: Requer MÚLTIPLOS critérios fortes
        if score >= 95 and cv < 0.50 and (has_strong_periodicity or hourly_conc > 60):
            classification = "🔴 REINCIDÊNCIA CRÍTICA (R1)"
        
        # R2: Requer critérios moderados + baixo cluster ratio
        elif score >= 70 and cv < 0.70 and cluster_ratio < 2.0:
            classification = "🟠 REINCIDÊNCIA PARCIAL (R2)"
        
        # R3: Sinais fracos mas detectáveis
        elif score >= 40 and cv < 1.2:
            classification = "🟡 REINCIDÊNCIA DETECTÁVEL (R3)"
        
        # R4: Não atende critérios
        else:
            classification = "🟢 NÃO REINCIDENTE (R4)"
        
        # Log detalhado
        reasons_text = "\n".join(reasons)
        print(f"\n{'='*70}")
        print(f"Alert: {self.alert_id}")
        print(f"Eventos: {total_events} | Cluster Ratio: {cluster_ratio:.2f}x")
        print(f"Score: {score} | CV: {cv:.3f}")
        print(f"Classificação: {classification}")
        print(f"\nMotivos:\n{reasons_text}")
        print(f"{'='*70}\n")
        
        return score, classification

    def _final_classification(self, results, df_events, intervals_hours, total_events, cluster_ratio):
        """Exibe classificação final no Streamlit."""
        score, classification = self._calculate_final_score_validated(
            results, df_events, intervals_hours, total_events, cluster_ratio
        )
        
        st.markdown("---")
        st.header("🎯 Classificação Final")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.metric("📊 Score Final", f"{score}/137")
            st.markdown(f"### {classification}")
            
            st.markdown("**Critérios Avaliados:**")
            cv = results['regularity'].get('cv', float('inf'))
            has_periodicity = results['periodicity'].get('has_strong_periodicity', False)
            hourly_conc = results['temporal'].get('hourly_concentration', 0)
            predictability = results['predictability'].get('predictability_score', 0)
            
            st.write(f"• **Eventos:** {total_events}")
            st.write(f"• **Cluster Ratio:** {cluster_ratio:.2f}x")
            st.write(f"• **Regularidade (CV):** {cv:.3f}")
            st.write(f"• **Periodicidade Forte:** {'✅ Sim' if has_periodicity else '❌ Não'}")
            st.write(f"• **Concentração Horária:** {hourly_conc:.1f}%")
            st.write(f"• **Previsibilidade:** {predictability:.0f}%")
        
        with col2:
            if score >= 95:
                color = "red"
            elif score >= 70:
                color = "orange"
            elif score >= 40:
                color = "yellow"
            else:
                color = "green"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score"},
                gauge={
                    'axis': {'range': [0, 137]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 95], 'color': "lightsalmon"},
                        {'range': [95, 137], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recomendações
        st.markdown("---")
        st.subheader("💡 Recomendações")
        
        if cluster_ratio >= 3.0:
            st.error("""
            **🚨 ATENÇÃO: CLUSTERS DOMINANTES**
            - Este não é um padrão de reincidência temporal
            - É um problema único gerando múltiplos alertas simultâneos
            - Investigar causa raiz do BURST de alertas
            - Considerar ajuste de threshold/correlação de alertas
            """)
        elif score >= 95:
            st.error("""
            **🔴 REINCIDÊNCIA CRÍTICA:**
            - Padrão temporal FORTE e previsível
            - Alta probabilidade de novas ocorrências
            - Requer análise de causa raiz IMEDIATA
            - Considerar automação de resposta
            """)
        elif score >= 70:
            st.warning("""
            **🟠 REINCIDÊNCIA PARCIAL:**
            - Padrão temporal MODERADO identificado
            - Alguns aspectos são previsíveis
            - Investigar causas recorrentes
            - Monitorar de perto
            """)
        elif score >= 40:
            st.info("""
            **🟡 REINCIDÊNCIA DETECTÁVEL:**
            - Sinais fracos de padrão temporal
            - Padrão ainda não consolidado
            - Continuar monitorando evolução
            - Documentar ocorrências
            """)
        else:
            st.success("""
            **🟢 NÃO REINCIDENTE:**
            - Não há padrão forte de reincidência
            - Ocorrências parecem independentes
            - Manutenção de monitoramento padrão
            """)

    def _analyze_basic_statistics(self, intervals, render=True):
        """Análise estatística básica dos intervalos entre eventos."""
        stats_dict = {
            'mean': float(np.mean(intervals)),
            'median': float(np.median(intervals)),
            'std': float(np.std(intervals)),
            'min': float(np.min(intervals)),
            'max': float(np.max(intervals)),
            'cv': float(np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')),
            'q25': float(np.percentile(intervals, 25)),
            'q75': float(np.percentile(intervals, 75)),
            'iqr': float(np.percentile(intervals, 75) - np.percentile(intervals, 25))
        }
        
        if render:
            st.subheader("📊 Estatísticas de Intervalos ENTRE Eventos")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("⏱️ Média", f"{stats_dict['mean']:.1f}h")
            col2.metric("📊 Mediana", f"{stats_dict['median']:.1f}h")
            col3.metric("📈 Desvio", f"{stats_dict['std']:.1f}h")
            col4.metric("⚡ Mínimo", f"{stats_dict['min']:.1f}h")
            col5.metric("🐌 Máximo", f"{stats_dict['max']:.1f}h")
        
        return stats_dict

    def _analyze_regularity(self, intervals, render=True):
        """Análise de regularidade usando CV robusto."""
        mediana = np.median(intervals)
        mad = np.median(np.abs(intervals - mediana))
        cv = mad / mediana if mediana > 0 else float('inf')

        if cv < 0.25:
            regularity_score, pattern_type, pattern_color = 100, "🟢 MUITO REGULAR", "green"
        elif cv < 0.40:
            regularity_score, pattern_type, pattern_color = 85, "🟢 REGULAR", "lightgreen"
        elif cv < 0.60:
            regularity_score, pattern_type, pattern_color = 65, "🟡 SEMI-REGULAR", "yellow"
        elif cv < 1.0:
            regularity_score, pattern_type, pattern_color = 35, "🟠 IRREGULAR", "orange"
        else:
            regularity_score, pattern_type, pattern_color = 10, "🔴 MUITO IRREGULAR", "red"

        if render:
            st.subheader("🎯 1. Regularidade (CV Robusto)")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Classificação:** {pattern_type}")
                st.write(f"**CV Robusto:** {cv:.3f}")
                if len(intervals) >= 3:
                    try:
                        _, p_value = stats.shapiro(intervals)
                        if p_value > 0.05:
                            st.info("📊 Distribuição aproximadamente normal")
                        else:
                            st.warning("📊 Distribuição não-normal")
                    except:
                        pass
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=regularity_score,
                    title={'text': "Regularidade"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': pattern_color}}
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True, key=f'reg_gauge_{self.alert_id}')

        return {'cv': cv, 'regularity_score': regularity_score, 'type': pattern_type}

    def _analyze_periodicity(self, intervals, render=True):
        """Detecção de periodicidade usando FFT."""
        if len(intervals) < 10:
            if render:
                st.subheader("🔍 2. Periodicidade (FFT)")
                st.info("📊 Mínimo de 10 intervalos necessários")
            return {
                'periods': [],
                'has_periodicity': False,
                'has_strong_periodicity': False,
                'has_moderate_periodicity': False,
                'dominant_period_hours': None
            }

        intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
        n_padded = 2**int(np.ceil(np.log2(len(intervals_norm))))
        intervals_padded = np.pad(intervals_norm, (0, n_padded - len(intervals_norm)), 'constant')

        fft_vals = fft(intervals_padded)
        freqs = fftfreq(n_padded, d=1)

        positive_idx = freqs > 0
        freqs_pos = freqs[positive_idx]
        fft_mag = np.abs(fft_vals[positive_idx])

        strong_threshold = np.mean(fft_mag) + 2 * np.std(fft_mag)
        moderate_threshold = np.mean(fft_mag) + np.std(fft_mag)

        strong_peaks_idx = fft_mag > strong_threshold
        moderate_peaks_idx = (fft_mag > moderate_threshold) & (fft_mag <= strong_threshold)

        dominant_periods = []
        has_strong_periodicity = False
        has_moderate_periodicity = False
        dominant_period_hours = None

        if np.any(strong_peaks_idx):
            dominant_freqs = freqs_pos[strong_peaks_idx]
            dominant_periods = (1 / dominant_freqs)
            dominant_periods = dominant_periods[dominant_periods < len(intervals)][:3]
            if len(dominant_periods) > 0:
                has_strong_periodicity = True
                dominant_period_hours = float(dominant_periods[0] * np.mean(intervals))

        if not has_strong_periodicity and np.any(moderate_peaks_idx):
            has_moderate_periodicity = True

        if render:
            st.subheader("🔍 2. Periodicidade (FFT)")
            if has_strong_periodicity:
                st.success("🎯 **Periodicidades Fortes Detectadas:**")
                for period in dominant_periods:
                    est_time = period * np.mean(intervals)
                    time_str = f"{est_time:.1f}h" if est_time < 24 else f"{est_time/24:.1f} dias"
                    st.write(f"• Período: **{period:.1f}** ocorrências (~{time_str})")
            elif has_moderate_periodicity:
                st.info("📊 **Periodicidade Moderada Detectada**")
            else:
                st.info("📊 Nenhuma periodicidade detectada")

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
            st.plotly_chart(fig, use_container_width=True, key=f'fft_{self.alert_id}')

        return {
            'periods': list(map(float, dominant_periods)) if len(dominant_periods) else [],
            'has_periodicity': len(dominant_periods) > 0,
            'has_strong_periodicity': has_strong_periodicity,
            'has_moderate_periodicity': has_moderate_periodicity,
            'dominant_period_hours': dominant_period_hours
        }

    def _analyze_temporal_patterns(self, df, render=True):
        """Análise de concentração temporal."""
        hourly = df.groupby('hour').size().reindex(range(24), fill_value=0)
        daily = df.groupby('day_of_week').size().reindex(range(7), fill_value=0)
        
        hourly_pct = (hourly / hourly.sum() * 100) if hourly.sum() > 0 else pd.Series(dtype=float)
        daily_pct = (daily / daily.sum() * 100) if daily.sum() > 0 else pd.Series(dtype=float)
        
        hourly_conc = float(hourly_pct.nlargest(3).sum()) if len(hourly_pct) > 0 else 0.0
        daily_conc = float(daily_pct.nlargest(3).sum()) if len(daily_pct) > 0 else 0.0
        
        peak_hours = hourly[hourly > hourly.mean() + hourly.std()].index.tolist() if len(hourly) > 0 else []
        peak_days = daily[daily > daily.mean() + daily.std()].index.tolist() if len(daily) > 0 else []

        if render:
            st.subheader("⏰ 3. Padrões Temporais")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(go.Bar(
                    x=list(range(24)), 
                    y=hourly.values,
                    marker_color=['red' if v > hourly.mean() + hourly.std() else 'lightblue' for v in hourly.values]
                ))
                fig.update_layout(title="Distribuição por Hora", xaxis_title="Hora", height=250)
                st.plotly_chart(fig, use_container_width=True, key=f'hourly_{self.alert_id}')
                if peak_hours:
                    st.success(f"🕐 **Picos:** {', '.join([f'{h:02d}:00' for h in peak_hours])}")
            
            with col2:
                days_map = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
                fig = go.Figure(go.Bar(
                    x=days_map, 
                    y=daily.values,
                    marker_color=['red' if v > daily.mean() + daily.std() else 'lightgreen' for v in daily.values]
                ))
                fig.update_layout(title="Distribuição por Dia", xaxis_title="Dia", height=250)
                st.plotly_chart(fig, use_container_width=True, key=f'daily_{self.alert_id}')
                if peak_days:
                    st.success(f"📅 **Picos:** {', '.join([days_map[d] for d in peak_days])}")

        return {
            'hourly_concentration': hourly_conc,
            'daily_concentration': daily_conc,
            'peak_hours': peak_hours,
            'peak_days': peak_days
        }

    def _calculate_predictability(self, intervals, render=True):
        """Calcula previsibilidade baseada em regressão linear."""
        if len(intervals) < 3:
            return {'predictability_score': 0, 'next_expected_hours': 0}

        X = np.arange(len(intervals)).reshape(-1, 1)
        y = intervals

        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            predictability_score = max(0, min(100, r2 * 100))
            next_expected = model.predict([[len(intervals)]])[0]
            
            if render:
                st.subheader("🔮 4. Previsibilidade")
                col1, col2 = st.columns(2)
                col1.metric("Score", f"{predictability_score:.0f}%")
                col2.metric("Próxima em", f"{next_expected:.1f}h")
            
            return {
                'predictability_score': float(predictability_score),
                'next_expected_hours': float(next_expected)
            }
        except:
            return {'predictability_score': 0, 'next_expected_hours': 0}

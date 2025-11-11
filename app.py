import pandas as pd
import numpy as np
from datetime import datetime


class ResultsComparator:
    """Compara resultados entre Athena e a pipeline de análise"""
    
    def __init__(self):
        self.athena_results = None
        self.pipeline_results = None
        self.comparison_report = None
    
    def load_athena_results(self, athena_df):
        """Carrega resultados do Athena"""
        self.athena_results = athena_df.copy()
        if 'is_reincidente' not in self.athena_results.columns:
            raise ValueError("DataFrame do Athena deve ter coluna 'is_reincidente'")
        self.athena_results['is_reincidente'] = self.athena_results['is_reincidente'].astype(bool)
        return self
    
    def load_pipeline_results(self, pipeline_df):
        """Carrega resultados da pipeline"""
        self.pipeline_results = pipeline_df.copy()
        # R1 e R2 = reincidente, R3 e R4 = não reincidente
        self.pipeline_results['is_reincidente'] = self.pipeline_results['classification'].apply(
            lambda x: 'R1' in str(x) or 'R2' in str(x) or 'CRÍTICO' in str(x) or 'PARCIALMENTE' in str(x)
        )
        return self
    
    def compare(self):
        """Realiza a comparação entre Athena e Pipeline"""
        if self.athena_results is None or self.pipeline_results is None:
            raise ValueError("Carregue ambos os resultados antes de comparar")
        
        comparison = pd.merge(
            self.athena_results[['u_alert_id', 'is_reincidente']],
            self.pipeline_results[['u_alert_id', 'is_reincidente', 'score', 'classification']],
            on='u_alert_id',
            how='outer',
            suffixes=('_athena', '_pipeline')
        )
        
        comparison['is_reincidente_athena'] = comparison['is_reincidente_athena'].fillna(False)
        comparison['is_reincidente_pipeline'] = comparison['is_reincidente_pipeline'].fillna(False)
        comparison['match_type'] = comparison.apply(self._categorize_match, axis=1)
        
        metrics = self._calculate_metrics(comparison)
        
        self.comparison_report = {
            'comparison_df': comparison,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.comparison_report
    
    def _categorize_match(self, row):
        """Categoriza o tipo de match entre Athena e Pipeline"""
        athena = row['is_reincidente_athena']
        pipeline = row['is_reincidente_pipeline']
        
        if athena and pipeline:
            return '✅ Concordam - Reincidente'
        elif not athena and not pipeline:
            return '✅ Concordam - Não Reincidente'
        elif athena and not pipeline:
            return '⚠️ Divergem - Athena: Sim, Pipeline: Não'
        else:
            return '⚠️ Divergem - Athena: Não, Pipeline: Sim'
    
    def _calculate_metrics(self, comparison_df):
        """Calcula métricas de comparação"""
        total = len(comparison_df)
        athena_reincidentes = comparison_df['is_reincidente_athena'].sum()
        pipeline_reincidentes = comparison_df['is_reincidente_pipeline'].sum()
        
        true_positives = ((comparison_df['is_reincidente_athena'] == True) & 
                         (comparison_df['is_reincidente_pipeline'] == True)).sum()
        true_negatives = ((comparison_df['is_reincidente_athena'] == False) & 
                         (comparison_df['is_reincidente_pipeline'] == False)).sum()
        false_positives = ((comparison_df['is_reincidente_athena'] == False) & 
                          (comparison_df['is_reincidente_pipeline'] == True)).sum()
        false_negatives = ((comparison_df['is_reincidente_athena'] == True) & 
                          (comparison_df['is_reincidente_pipeline'] == False)).sum()
        
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        agreement_rate = (true_positives + true_negatives) / total if total > 0 else 0
        
        return {
            'total_alertas': int(total),
            'athena_reincidentes': int(athena_reincidentes),
            'pipeline_reincidentes': int(pipeline_reincidentes),
            'concordancias': int(true_positives + true_negatives),
            'divergencias': int(false_positives + false_negatives),
            'agreement_rate': float(agreement_rate),
            'confusion_matrix': {
                'true_positives': int(true_positives),
                'true_negatives': int(true_negatives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives)
            },
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score)
            }
        }
    
    def get_summary(self):
        """Retorna resumo formatado da comparação"""
        if self.comparison_report is None:
            return "Nenhuma comparação realizada ainda."
        
        metrics = self.comparison_report['metrics']
        
        return f"""
📊 COMPARAÇÃO ATHENA vs PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 Totais:
  • Total de Alertas: {metrics['total_alertas']}
  • Reincidentes (Athena): {metrics['athena_reincidentes']} ({metrics['athena_reincidentes']/metrics['total_alertas']*100:.1f}%)
  • Reincidentes (Pipeline): {metrics['pipeline_reincidentes']} ({metrics['pipeline_reincidentes']/metrics['total_alertas']*100:.1f}%)

🎯 Concordância:
  • Concordam: {metrics['concordancias']} ({metrics['agreement_rate']*100:.1f}%)
  • Divergem: {metrics['divergencias']} ({(1-metrics['agreement_rate'])*100:.1f}%)

📊 Matriz de Confusão:
  • True Positives:  {metrics['confusion_matrix']['true_positives']}
  • True Negatives:  {metrics['confusion_matrix']['true_negatives']}
  • False Positives: {metrics['confusion_matrix']['false_positives']}
  • False Negatives: {metrics['confusion_matrix']['false_negatives']}

📏 Métricas de Performance:
  • Accuracy:  {metrics['metrics']['accuracy']*100:.2f}%
  • Precision: {metrics['metrics']['precision']*100:.2f}%
  • Recall:    {metrics['metrics']['recall']*100:.2f}%
  • F1-Score:  {metrics['metrics']['f1_score']*100:.2f}%
        """
    
    def export_to_dict(self):
        """Exporta comparação para dicionário"""
        if self.comparison_report is None:
            return None
        
        return {
            'metrics': self.comparison_report['metrics'],
            'timestamp': self.comparison_report['timestamp'],
            'comparison_data': self.comparison_report['comparison_df'].to_dict('records')
        }
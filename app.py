import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st


class DataStorage:
    """Gerencia persistência de dados de análise"""
    
    def __init__(self, storage_dir="./data_cache"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.athena_cache_file = self.storage_dir / "athena_results.json"
        self.pipeline_cache_file = self.storage_dir / "pipeline_results.json"
        self.comparison_cache_file = self.storage_dir / "comparison_results.json"
        self.metadata_file = self.storage_dir / "metadata.json"
    
    def save_athena_results(self, df):
        """Salva resultados do Athena"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'data': df.to_dict('records'),
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'columns': list(df.columns)
        }
        
        with open(self.athena_cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        self._update_metadata('athena', data['timestamp'])
        return True
    
    def load_athena_results(self):
        """Carrega resultados do Athena do cache"""
        if not self.athena_cache_file.exists():
            return None
        
        try:
            with open(self.athena_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data['data'])
            if 'created_on' in df.columns:
                df['created_on'] = pd.to_datetime(df['created_on'])
            
            return df, data['timestamp']
        except Exception as e:
            print(f"Erro ao carregar cache do Athena: {e}")
            return None
    
    def save_pipeline_results(self, df):
        """Salva resultados da pipeline"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'data': df.to_dict('records'),
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'summary': {
                'total': len(df),
                'r1_count': len(df[df['classification'].str.contains('R1|CRÍTICO', na=False)]),
                'r2_count': len(df[df['classification'].str.contains('R2|PARCIALMENTE', na=False)]),
                'r3_count': len(df[df['classification'].str.contains('R3|DETECTÁVEL', na=False)]),
                'r4_count': len(df[df['classification'].str.contains('R4|NÃO', na=False)])
            }
        }
        
        with open(self.pipeline_cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        self._update_metadata('pipeline', data['timestamp'])
        return True
    
    def load_pipeline_results(self):
        """Carrega resultados da pipeline do cache"""
        if not self.pipeline_cache_file.exists():
            return None
        
        try:
            with open(self.pipeline_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data['data'])
            return df, data['timestamp'], data.get('summary', {})
        except Exception as e:
            print(f"Erro ao carregar cache da pipeline: {e}")
            return None
    
    def save_comparison_results(self, comparison_data):
        """Salva resultados da comparação"""
        with open(self.comparison_cache_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        self._update_metadata('comparison', datetime.now().isoformat())
        return True
    
    def load_comparison_results(self):
        """Carrega resultados da comparação do cache"""
        if not self.comparison_cache_file.exists():
            return None
        
        try:
            with open(self.comparison_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Erro ao carregar cache da comparação: {e}")
            return None
    
    def _update_metadata(self, data_type, timestamp):
        """Atualiza metadados do cache"""
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        metadata[data_type] = {
            'last_updated': timestamp,
            'file_size': os.path.getsize(getattr(self, f'{data_type}_cache_file'))
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def get_metadata(self):
        """Retorna metadados do cache"""
        if not self.metadata_file.exists():
            return {}
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def has_cached_data(self):
        """Verifica se existe dados em cache"""
        return {
            'athena': self.athena_cache_file.exists(),
            'pipeline': self.pipeline_cache_file.exists(),
            'comparison': self.comparison_cache_file.exists()
        }
    
    def clear_cache(self, data_type=None):
        """Limpa cache (específico ou todos)"""
        if data_type:
            cache_file = getattr(self, f'{data_type}_cache_file', None)
            if cache_file and cache_file.exists():
                cache_file.unlink()
        else:
            for file in self.storage_dir.glob('*.json'):
                file.unlink()
        return True
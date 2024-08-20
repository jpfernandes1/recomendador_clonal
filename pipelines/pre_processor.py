from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from statistics import mode
import pandas as pd
import numpy as np

class OutlierRemoverIQR(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Para cada coluna numérica, remover outliers usando o método IQR
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (self.factor * IQR)
                upper_bound = Q3 + (self.factor * IQR)
                X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
        return X

def substituir_valores_raros(df, limite=10, excluir_colunas=[]):
    """
    Substitui valores raros em colunas do tipo object por 'Outros', com base em um limite especificado.

    Parâmetros:
    df (pd.DataFrame): DataFrame de entrada.
    limite (int): O limite mínimo de contagem para manter o valor original. Valores abaixo deste serão substituídos por 'Outros'.
    excluir_colunas (list): Lista de colunas do tipo object que não devem ser afetadas.

    Retorno:
    pd.DataFrame: DataFrame com os valores substituídos conforme a condição.
    """
    
    # Iterando pelas colunas do DataFrame que são do tipo 'object'
    for col in df.select_dtypes(include='object').columns:
        if col not in excluir_colunas:
            # Calculando as contagens de cada valor na coluna
            value_counts = df[col].value_counts()
            
            # Substituindo valores raros por 'Outros'
            df[col] = df[col].apply(lambda x: x if value_counts[x] >= limite else 'Outros')
    
    return df
# Exemplo de uso:
# substituir_valores_raros(df, limite=10, excluir_colunas=['UP'])


def calcular_arvores_por_ha(df, coluna):
    # Obtendo os valores da coluna
    valores_conhecidos = df[coluna].apply(lambda x: x if 'x' in x else None).dropna()
    # Obtendo a moda dos valores reconhecidos na coluna
    moda_valores = mode(valores_conhecidos)
    
    def converter_espacamento(espacamento):
        try:
            # Verifica se o layout está no formato "NxM"
            if 'x' in espacamento:
                # Converte a string espaçamento (ex.:"300x300") em dois valores inteiros
                largura, comprimento = map(int, espacamento.split('x'))
                
                # Converte centímetros para metros
                largura_metros = largura / 100
                comprimento_metros = comprimento / 100
                
                # Calcula a área de uma árvore
                area_por_arvore = largura_metros * comprimento_metros
                
                # Calcula o número de árvores por hectare (10.000 m²)
                arvores_por_ha = 10000 / area_por_arvore
                
                return round(arvores_por_ha)
            else:
                # Caso o formato não seja reconhecido, retorna a moda
                return converter_espacamento(moda_valores)
        except ValueError:
            # Caso haja um erro na conversão ou no formato, retorna a moda
            return converter_espacamento(moda_valores)

    # Aplicando a conversão na coluna específica do DataFrame
    df[coluna] = df[coluna].apply(converter_espacamento)
    df.rename(columns={coluna: 'arv/ha'}, inplace = True)

# Exemplo de uso:
# calcular_arvores_por_ha(seu_dataframe, 'coluna_de_espacamento', 'arv/ha')


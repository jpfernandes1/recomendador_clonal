import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from statistics import mode


class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()  # Cria uma cópia para não alterar o DataFrame original
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (self.factor * iqr)
            upper_bound = q3 + (self.factor * iqr)
            X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
        return X

def create_pipeline(df):
    # Identifica colunas numéricas e categóricas automaticamente
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # Cria um pipeline para processamento das variáveis categóricas e numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('remove_outliers', RemoveOutliers()),
                ('scaler', MinMaxScaler())
            ]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Ajusta o pipeline aos dados e retorna
    pipeline.fit(df)
    
    return pipeline

# Exemplo de como executar:
'''# Importa a função do arquivo pre_processor.py
from pre_processor import create_pipeline

# Suponha que df seja o seu DataFrame
df = pd.read_csv('seu_arquivo.csv')  # Substitua com o caminho do seu arquivo

# Cria o pipeline com o DataFrame
pipeline = create_pipeline(df)

# Transforme os dados de treino e teste
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)'''


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


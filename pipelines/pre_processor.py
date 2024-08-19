from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from statistics import mode
import pandas as pd
import numpy as np

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3.0):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        numeric_cols = X_out.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X_out = X_out[(np.abs(X_out[col] - X_out[col].mean()) / X_out[col].std() < self.threshold)]
        return X_out

def create_preprocessing_pipeline(df):
    # Identifica variáveis numéricas e categóricas automaticamente
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns

    # Define o pré-processamento para variáveis numéricas e categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('outlier_remover', OutlierRemover()),  # Remove outliers
                ('scaler', StandardScaler())  # Aplica Scaler
            ]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Aplica OneHotEncoder
        ])

    return preprocessor

def create_model_pipeline():
    # Cria a pipeline com o modelo de regressão
    pipeline = Pipeline(steps=[
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    return pipeline

def train_and_evaluate_model(model_pipeline, df, target_column):
    # Divide os dados em X e y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Treina o modelo com os dados completos
    model_pipeline.fit(X, y)

    # Faz previsões nos próprios dados (não ideal, mas necessário dado o cenário)
    y_pred = model_pipeline.predict(X)

    # Avalia o modelo
    r2 = r2_score(y, y_pred)
    
    # Exibir as métricas de avaliação
    print(f"R²: {r2}")

    return model_pipeline

"""
# Supondo que você já tenha os dados carregados no DataFrame `df`
# E a coluna `target_column` representa a coluna que você está tentando prever (por exemplo, produtividade)

# Crie e aplique a pipeline de pré-processamento
preprocessor = create_preprocessing_pipeline(df)
df_processed = apply_preprocessing(preprocessor, df)

# Crie e treine a pipeline de modelo
model_pipeline = create_model_pipeline()
trained_model = train_and_evaluate_model(model_pipeline, df_processed, target_column='produtividade')
"""

def apply_preprocessing(preprocessor, df):
    # Ajusta a pipeline aos dados e transforma o conjunto de dados
    df_transformed = preprocessor.fit_transform(df)
    
    # Retorna o DataFrame transformado
    df_transformed_df = pd.DataFrame(df_transformed, columns=preprocessor.get_feature_names_out())
    
    return df_transformed_df


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


import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

# Função para treinar o modelo e salvar o preprocessador e o modelo treinado em disco
def train_and_save_model(data, model_path, preprocessor_path):
    # Separando as variáveis independentes (X) e dependente (y)
    X = data.drop(columns=['avg_vol/ha'])  # Dados sem a variável alvo
    y = data['avg_vol/ha']  # Variável alvo

    # Identificando colunas categóricas e numéricas
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Criando o pré-processador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Criando o pipeline completo (pré-processamento + modelo)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Treinando o pipeline
    pipeline.fit(X, y)

    # Salvando o modelo treinado, o preprocessador, e o label encoder em disco
    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline.named_steps['preprocessor'], preprocessor_path)

    print(f"Modelo salvo em: {model_path}")
    print(f"Preprocessador salvo em: {preprocessor_path}")


def predict_and_recommend(pipeline_path, preprocessor_path, new_data, genetic_materials):
    # Carregando o pipeline e o label encoder
    pipeline = joblib.load(pipeline_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Identificando colunas categóricas e numéricas
    numeric_features = new_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = new_data.select_dtypes(include=['object']).columns

    # Criando uma lista para armazenar os resultados
    resultados = []

    # Criando o novo data frame
    data_with_material = new_data.copy()

    # Iterando sobre os materiais genéticos possíveis
    for material in genetic_materials:
        # Criando uma cópia dos dados e inserindo o material genético
        data_with_material['Material Genético'] = material
        # Aplicando o modelo para prever a produtividade
        produtividade_prevista = pipeline.predict(data_with_material)

        # Adicionando à lista a previsão para o material
        resultados.append({
            'Material Genético': material,
            'Produtividade Prevista': produtividade_prevista[0]
        })

    # Transformando resultado em um Data Frame
    resultados = pd.DataFrame(resultados)

    # Ordenando os resultados pela produtividade prevista em ordem decrescente
    results_sorted = resultados.sort_values(by='Produtividade Prevista', ascending=False)
    
    # Retornando os três materiais genéticos com maior produtividade prevista
    return results_sorted[:3]





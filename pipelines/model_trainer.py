import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

# Função para treinar o modelo e salvar o preprocessador e o modelo treinado em disco
def train_and_save_model(data, target_col, model_path, preprocessor_path, label_encoder_path):
    # Separando as variáveis independentes (X) e dependente (y)
    X = data.drop(columns=[target_col])  # Dados sem a variável alvo
    y = data[target_col]  # Variável alvo

    # Identificando colunas categóricas e numéricas
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Encodando a variável alvo (Material Genético) para treinamento
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

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
    pipeline.fit(X, y_encoded)

    # Salvando o modelo treinado, o preprocessador, e o label encoder em disco
    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(pipeline.named_steps['preprocessor'], preprocessor_path)

    print(f"Modelo salvo em: {model_path}")
    print(f"Preprocessador salvo em: {preprocessor_path}")
    print(f"Label Encoder salvo em: {label_encoder_path}")


def predict_and_recommend(pipeline_path, label_encoder_path, preprocessor_path, new_data, genetic_materials):
    # Carregando o pipeline e o label encoder
    pipeline = joblib.load(pipeline_path)
    label_encoder = joblib.load(label_encoder_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Criando uma lista para armazenar os resultados
    resultados = []

    # Identificando colunas categóricas e numéricas
    numeric_features = new_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = new_data.select_dtypes(include=['object']).columns

    # Criando o novo data frame
    data_with_material = new_data.copy()

    # Iterando sobre os materiais genéticos possíveis
    for material in genetic_materials:
        # Criando uma cópia dos dados e inserindo o material genético
        data_with_material['Material Genético'] = material

        # Aplicando o preprocessador (minmaxscaler, onehotencoder)
        data_preprocessed = preprocessor.transform(data_with_material)

        # Aplicando o modelo para prever a produtividade
        produtividade_prevista = pipeline.predict(data_preprocessed)
        
        # Decodificando as previsões para strings
        resultados.append({
            'Material Genético': material,
            'Produtividade Prevista': produtividade_prevista[0]
        })

    # Ordenando os resultados pela produtividade prevista em ordem decrescente
    results_sorted = sorted(resultados, key=lambda x: x[1], reverse=True)
    
    # Retornando os três materiais genéticos com maior produtividade prevista
    return results_sorted[:3]





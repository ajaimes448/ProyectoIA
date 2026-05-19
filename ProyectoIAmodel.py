# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import kagglehub
from kagglehub import KaggleDatasetAdapter
import joblib
import os

# Configuración
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_and_preprocess_data():
    """Carga el dataset desde kagglehub y aplica preprocesamiento básico"""
    # Cargar datos
    file_path = "banking_transactions.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "deepeshkansotia/banking-fraud-detection-and-risk-analytics-dataset",
        file_path,
    )
    
    # Eliminar outliers de anomaly_score
    Q1 = df['anomaly_score'].quantile(0.25)
    Q3 = df['anomaly_score'].quantile(0.75)
    iqr = Q3 - Q1
    lower_bound = Q1 - 1.5 * iqr
    upper_bound = Q3 + 1.5 * iqr
    df_clip = df.copy()
    df_clip['anomaly_score'] = df_clip['anomaly_score'].clip(lower_bound, upper_bound)
    df['anomaly_score'] = df_clip['anomaly_score']
    
    # Codificar variables categóricas
    encoder = LabelEncoder()
    categorical = df.select_dtypes(include=['object']).columns
    for col in categorical:
        df[col] = encoder.fit_transform(df[col])
    
    # Escalar variables numéricas
    scaler = StandardScaler()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical:
        df[col] = scaler.fit_transform(df[[col]])
    
    return df, scaler, encoder

def prepare_features(df):
    """Prepara X e y para el modelo"""
    X = df.drop(columns=['fraud_flag'])
    y = df['fraud_flag']
    return X, y

def apply_pca(X, n_components=17):
    """Aplica PCA a las características numéricas"""
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X[numerical_cols])
    return X_pca, pca

def apply_lda(X, y, n_components=1):
    """Aplica LDA a las características numéricas"""
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    lda = LDA(n_components=n_components)
    X_lda = lda.fit_transform(X[numerical_cols], y)
    return X_lda, lda

def get_categorical_features(X):
    """Obtiene las características categóricas"""
    categorical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    # Nota: Después de codificar, las categóricas también son numéricas
    # Por simplicidad, usamos todas las columnas
    return X

def train_model(X, y, model_type='original'):
    """Entrena el modelo Random Forest"""
    # Dividir datos
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Configurar y entrenar modelo
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    model.fit(x_train, y_train)
    
    return model, x_test, y_test

def train_models_pipeline(df):
    """Entrena los modelos PCA, LDA y original"""
    X, y = prepare_features(df)
    
    # Obtener características categóricas
    categorical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Modelo Original
    model_orig, x_test_orig, y_test_orig = train_model(X, y, 'original')
    
    # Modelo PCA
    X_pca, pca = apply_pca(X)
    X_pca_combined = np.concatenate((X_pca, X[categorical_cols]), axis=1)
    model_pca, x_test_pca, y_test_pca = train_model(X_pca_combined, y, 'pca')
    
    # Modelo LDA
    X_lda, lda = apply_lda(X, y)
    X_lda_combined = np.concatenate((X_lda, X[categorical_cols]), axis=1)
    model_lda, x_test_lda, y_test_lda = train_model(X_lda_combined, y, 'lda')
    
    return {
        'model_orig': model_orig,
        'model_pca': model_pca,
        'model_lda': model_lda,
        'pca': pca,
        'lda': lda,
        'x_test_orig': x_test_orig,
        'y_test_orig': y_test_orig,
        'x_test_pca': x_test_pca,
        'y_test_pca': y_test_pca,
        'x_test_lda': x_test_lda,
        'y_test_lda': y_test_lda,
        'categorical_cols': categorical_cols
    }

def save_models(models_dict, model_dir='models'):
    """Guarda los modelos entrenados en archivos"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    joblib.dump(models_dict['model_orig'], f'{model_dir}/random_forest_original.pkl')
    joblib.dump(models_dict['model_pca'], f'{model_dir}/random_forest_pca.pkl')
    joblib.dump(models_dict['model_lda'], f'{model_dir}/random_forest_lda.pkl')
    joblib.dump(models_dict['pca'], f'{model_dir}/pca.pkl')
    joblib.dump(models_dict['lda'], f'{model_dir}/lda.pkl')
    joblib.dump(models_dict['categorical_cols'], f'{model_dir}/categorical_cols.pkl')
    
    print(f"Modelos guardados en '{model_dir}/'")

def load_models(model_dir='models'):
    """Carga los modelos desde archivos"""
    model_orig = joblib.load(f'{model_dir}/random_forest_original.pkl')
    model_pca = joblib.load(f'{model_dir}/random_forest_pca.pkl')
    model_lda = joblib.load(f'{model_dir}/random_forest_lda.pkl')
    pca = joblib.load(f'{model_dir}/pca.pkl')
    lda = joblib.load(f'{model_dir}/lda.pkl')
    categorical_cols = joblib.load(f'{model_dir}/categorical_cols.pkl')
    
    return {
        'model_orig': model_orig,
        'model_pca': model_pca,
        'model_lda': model_lda,
        'pca': pca,
        'lda': lda,
        'categorical_cols': categorical_cols
    }

def predict_fraud(model_type, input_data, models_dict, scaler=None):
    """
    Realiza predicción de fraude
    
    Args:
        model_type: 'original', 'pca', o 'lda'
        input_data: Diccionario con los datos de entrada
        models_dict: Diccionario con modelos cargados
        scaler: Scaler para normalizar datos (opcional)
    
    Returns:
        Predicción (0 o 1) y probabilidad
    """
    # Convertir input_data a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Escalar si es necesario (esto requiere el scaler guardado)
    if scaler:
        numerical_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            input_df[col] = scaler.transform(input_df[[col]])
    
    # Seleccionar modelo
    if model_type == 'original':
        model = models_dict['model_orig']
        X_input = input_df
    elif model_type == 'pca':
        model = models_dict['model_pca']
        pca = models_dict['pca']
        categorical_cols = models_dict['categorical_cols']
        numerical = input_df.select_dtypes(include=['int64', 'float64']).columns
        X_pca = pca.transform(input_df[numerical])
        X_input = np.concatenate((X_pca, input_df[categorical_cols]), axis=1)
    elif model_type == 'lda':
        model = models_dict['model_lda']
        lda = models_dict['lda']
        categorical_cols = models_dict['categorical_cols']
        numerical = input_df.select_dtypes(include=['int64', 'float64']).columns
        X_lda = lda.transform(input_df[numerical])
        X_input = np.concatenate((X_lda, input_df[categorical_cols]), axis=1)
    else:
        raise ValueError("model_type debe ser 'original', 'pca' o 'lda'")
    
    # Predicción
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]
    
    return prediction, probability

# Alias para mantener compatibilidad con el notebook
model_pca = None
model_lda = None
model_X = None

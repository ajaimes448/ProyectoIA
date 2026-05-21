# model.py - Sistema de Detección de Fraude Bancario
# --- SÍNTESIS Y MODULARIZACIÓN ---

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE = 0.2

def generate_sample_data(n_samples=10000, random_state=42):
    """Genera datos de ejemplo para detección de fraude"""
    np.random.seed(random_state)
    data = {
        'transaction_amount': np.random.exponential(10000, n_samples),
        'login_attempts': np.random.poisson(2, n_samples),
        'device_risk_score': np.random.uniform(0, 100, n_samples),
        'transfer_frequency': np.random.poisson(30, n_samples),
        'anomaly_score': np.random.beta(2, 5, n_samples),
        'account_age_days': np.random.exponential(1000, n_samples),
        'transaction_time_hour': np.random.randint(0, 24, n_samples),
        'failed_transactions_last_30d': np.random.poisson(2, n_samples),
        'avg_monthly_balance': np.random.exponential(500000, n_samples),
        'daily_transaction_count': np.random.poisson(50, n_samples),
        'geo_distance_km': np.random.exponential(1000, n_samples),
        'session_duration_minutes': np.random.exponential(30, n_samples),
        'transaction_velocity_score': np.random.uniform(0, 100, n_samples),
        'payment_channel': np.random.choice(['ATM', 'Mobile App', 'POS Terminal', 'Web Banking'], n_samples),
        'authentication_type': np.random.choice(['Biometric', 'OTP', 'Password Only'], n_samples),
        'card_present_flag': np.random.choice([0, 1], n_samples),
        'international_transaction_flag': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'suspicious_ip_flag': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    df = pd.DataFrame(data)
    
    # Generar flag de fraude basado en reglas
    fraud_prob = (
        (df['anomaly_score'] > 0.7) * 0.3 +
        (df['login_attempts'] > 5) * 0.2 +
        (df['suspicious_ip_flag'] == 1) * 0.2 +
        (df['international_transaction_flag'] == 1) * 0.15 +
        (df['transaction_amount'] > 50000) * 0.15
    )
    df['fraud_flag'] = (np.random.random(n_samples) < fraud_prob).astype(int)
    
    # Tratamiento de outliers en anomaly_score
    Q1 = df['anomaly_score'].quantile(0.25)
    Q3 = df['anomaly_score'].quantile(0.75)
    iqr = Q3 - Q1
    lower_bound = Q1 - 1.5 * iqr
    upper_bound = Q3 + 1.5 * iqr
    df['anomaly_score'] = df['anomaly_score'].clip(lower_bound, upper_bound)
    
    return df

def preprocess_data(df):
    """Preprocesa los datos: codifica categóricas y escala numéricas"""
    df_encoded = df.copy()
    
    # Identificar y codificar variables categóricas
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    # Identificar y escalar variables numéricas
    numerical_cols = [col for col in df_encoded.select_dtypes(include=['int64', 'float64']).columns 
                     if col != 'fraud_flag']
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    return df_encoded, encoders, scaler, numerical_cols, categorical_cols

def train_models_pipeline(df):
    """
    Entrena los modelos de detección de fraude:
    - Random Forest original
    - Random Forest con PCA
    - Random Forest con LDA
    """
    # Preprocesar datos
    df_encoded, encoders, scaler, numerical_cols, categorical_cols = preprocess_data(df)
    
    # Separar features y target
    X = df_encoded.drop(columns=['fraud_flag'])
    y = df_encoded['fraud_flag']
    
    # Dividir en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # ==================== MODELO ORIGINAL ====================
    model_orig = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model_orig.fit(x_train, y_train)
    
    # ==================== MODELO CON PCA ====================
    n_components = min(10, len(numerical_cols))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca_train = pca.fit_transform(x_train[numerical_cols])
    X_pca_test = pca.transform(x_test[numerical_cols])
    
    # Combinar componentes PCA con variables categóricas
    categorical_train = x_train[categorical_cols].values if len(categorical_cols) > 0 else np.array([]).reshape(len(x_train), 0)
    categorical_test = x_test[categorical_cols].values if len(categorical_cols) > 0 else np.array([]).reshape(len(x_test), 0)
    
    if len(categorical_cols) > 0:
        X_pca_train_combined = np.concatenate([X_pca_train, categorical_train], axis=1)
        X_pca_test_combined = np.concatenate([X_pca_test, categorical_test], axis=1)
    else:
        X_pca_train_combined = X_pca_train
        X_pca_test_combined = X_pca_test
    
    model_pca = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', bootstrap=True,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    )
    model_pca.fit(X_pca_train_combined, y_train)
    
    # ==================== MODELO CON LDA ====================
    model_lda = None
    lda = None
    X_lda_test_combined = None
    
    try:
        if len(np.unique(y)) >= 2 and len(numerical_cols) > 1:
            lda = LDA(n_components=1)
            X_lda_train = lda.fit_transform(x_train[numerical_cols], y_train)
            X_lda_test = lda.transform(x_test[numerical_cols])
            
            if len(categorical_cols) > 0:
                X_lda_train_combined = np.concatenate([X_lda_train, categorical_train], axis=1)
                X_lda_test_combined = np.concatenate([X_lda_test, categorical_test], axis=1)
            else:
                X_lda_train_combined = X_lda_train
                X_lda_test_combined = X_lda_test
            
            model_lda = RandomForestClassifier(
                n_estimators=100, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', bootstrap=True,
                class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
            )
            model_lda.fit(X_lda_train_combined, y_train)
    except Exception as e:
        print(f"Error entrenando LDA: {e}")
        model_lda = None
        lda = None
        X_lda_test_combined = None
    
    # ==================== RETORNAR TODOS LOS MODELOS Y COMPONENTES ====================
    return {
        # Modelos
        'model': model_orig,
        'model_pca': model_pca,
        'model_lda': model_lda,
        
        # Transformadores
        'pca': pca,
        'lda': lda,
        'scaler': scaler,
        'encoders': encoders,
        
        # Metadata
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        
        # Datos de prueba - Modelo Original
        'x_test': x_test,
        'y_test': y_test,
        
        # Datos de prueba - PCA
        'x_test_pca': X_pca_test_combined,
        'y_test_pca': y_test,
        
        # Datos de prueba - LDA
        'x_test_lda': X_lda_test_combined,
        'y_test_lda': y_test if model_lda is not None else None,
        
        # Feature names completos (para consistencia)
        'feature_names': x_test.columns.tolist()
    }

def predict_transaction(models, transaction_data):
    """
    Predice si una nueva transacción es fraudulenta usando los modelos entrenados.
    
    Args:
        models: Diccionario retornado por train_models_pipeline
        transaction_data: Diccionario con los datos de la transacción
    
    Returns:
        Diccionario con predicciones y probabilidades
    """
    # Convertir a DataFrame
    input_df = pd.DataFrame([transaction_data])
    
    # Aplicar codificación a variables categóricas
    for col in models['categorical_cols']:
        if col in input_df.columns and col in models['encoders']:
            input_df[col] = models['encoders'][col].transform(input_df[col].astype(str))
    
    # Aplicar escalado a variables numéricas
    input_df[models['numerical_cols']] = models['scaler'].transform(input_df[models['numerical_cols']])
    
    # Asegurar el orden de las columnas
    input_df = input_df[models['feature_names']]
    
    # Predicciones
    pred_orig = models['model'].predict(input_df)[0]
    prob_orig = models['model'].predict_proba(input_df)[0][1]
    
    # Predicción con PCA
    X_pca = models['pca'].transform(input_df[models['numerical_cols']])
    if len(models['categorical_cols']) > 0:
        categorical_input = input_df[models['categorical_cols']].values
        X_pca_combined = np.concatenate([X_pca, categorical_input], axis=1)
    else:
        X_pca_combined = X_pca
    pred_pca = models['model_pca'].predict(X_pca_combined)[0]
    prob_pca = models['model_pca'].predict_proba(X_pca_combined)[0][1]
    
    # Predicción con LDA
    pred_lda = pred_orig
    prob_lda = prob_orig
    if models['model_lda'] is not None and models['lda'] is not None:
        X_lda = models['lda'].transform(input_df[models['numerical_cols']])
        if len(models['categorical_cols']) > 0:
            X_lda_combined = np.concatenate([X_lda, categorical_input], axis=1)
        else:
            X_lda_combined = X_lda
        pred_lda = models['model_lda'].predict(X_lda_combined)[0]
        prob_lda = models['model_lda'].predict_proba(X_lda_combined)[0][1]
    
    return {
        'original': {'prediction': int(pred_orig), 'probability': float(prob_orig)},
        'pca': {'prediction': int(pred_pca), 'probability': float(prob_pca)},
        'lda': {'prediction': int(pred_lda), 'probability': float(prob_lda)},
        'consensus': 1 if (pred_orig + pred_pca + pred_lda) >= 2 else 0
    }

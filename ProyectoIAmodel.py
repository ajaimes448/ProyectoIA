# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import kagglehub
import joblib
import os

# Configuración
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_and_preprocess_data(use_sample=True):
    """Carga el dataset desde kagglehub o genera datos de ejemplo"""
    if use_sample:
        # Generar datos de ejemplo (similar a app.py)
        np.random.seed(42)
        n_samples = 10000
        
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
        
        # Generar etiqueta de fraude
        fraud_prob = (
            (df['anomaly_score'] > 0.7) * 0.3 +
            (df['login_attempts'] > 5) * 0.2 +
            (df['suspicious_ip_flag'] == 1) * 0.2 +
            (df['international_transaction_flag'] == 1) * 0.15 +
            (df['transaction_amount'] > 50000) * 0.15
        )
        
        df['fraud_flag'] = (np.random.random(n_samples) < fraud_prob).astype(int)
        
        # Limitar outliers
        Q1 = df['anomaly_score'].quantile(0.25)
        Q3 = df['anomaly_score'].quantile(0.75)
        iqr = Q3 - Q1
        lower_bound = Q1 - 1.5 * iqr
        upper_bound = Q3 + 1.5 * iqr
        df['anomaly_score'] = df['anomaly_score'].clip(lower_bound, upper_bound)
        
        return df, StandardScaler(), LabelEncoder()
    
    else:
        # Cargar datos reales de Kaggle
        try:
            path = kagglehub.dataset_download("deepeshkansotia/banking-fraud-detection-and-risk-analytics-dataset")
            file_path = "banking_transactions.csv"
            df = pd.read_csv(os.path.join(path, file_path))
            
            # Limpiar outliers
            Q1 = df['anomaly_score'].quantile(0.25)
            Q3 = df['anomaly_score'].quantile(0.75)
            iqr = Q3 - Q1
            lower_bound = Q1 - 1.5 * iqr
            upper_bound = Q3 + 1.5 * iqr
            df['anomaly_score'] = df['anomaly_score'].clip(lower_bound, upper_bound)
            
            return df, StandardScaler(), LabelEncoder()
        except Exception as e:
            print(f"Error cargando datos de Kaggle: {e}")
            print("Usando datos de ejemplo como fallback")
            return load_and_preprocess_data(use_sample=True)

def prepare_features(df, scaler=None, encoder=None):
    """Prepara X e y para el modelo"""
    # Codificar variables categóricas si no se proporciona encoder
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if encoder is None:
        encoder = LabelEncoder()
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
    else:
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
    
    # Escalar variables numéricas
    numerical_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'fraud_flag']
    
    if scaler is None:
        scaler = StandardScaler()
        df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    else:
        df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
    
    X = df_encoded.drop(columns=['fraud_flag'])
    y = df_encoded['fraud_flag']
    
    return X, y, scaler, encoder, numerical_cols, categorical_cols

def train_models_pipeline(df):
    """Entrena los modelos PCA, LDA y original"""
    X, y, scaler, encoder, numerical_cols, categorical_cols = prepare_features(df)
    
    # Dividir datos
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Modelo Original
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
    
    # PCA
    n_components = min(10, len(numerical_cols))
    pca = PCA(n_components=n_components)
    X_pca_train = pca.fit_transform(x_train[numerical_cols])
    X_pca_test = pca.transform(x_test[numerical_cols])
    
    # Combinar PCA con características categóricas
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
    
    # LDA
    if len(np.unique(y)) >= 2:
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
    else:
        model_lda = None
        lda = None
        X_lda_test_combined = None
    
    return {
        'model_orig': model_orig,
        'model_pca': model_pca,
        'model_lda': model_lda,
        'pca': pca,
        'lda': lda,
        'scaler': scaler,
        'encoder': encoder,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'x_test': x_test,
        'y_test': y_test,
        'x_test_pca': X_pca_test_combined,
        'x_test_lda': X_lda_test_combined if model_lda else None
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
    joblib.dump(models_dict['scaler'], f'{model_dir}/scaler.pkl')
    joblib.dump(models_dict['encoder'], f'{model_dir}/encoder.pkl')
    joblib.dump(models_dict['numerical_cols'], f'{model_dir}/numerical_cols.pkl')
    joblib.dump(models_dict['categorical_cols'], f'{model_dir}/categorical_cols.pkl')
    
    print(f"Modelos guardados en '{model_dir}/'")

def load_models(model_dir='models'):
    """Carga los modelos desde archivos"""
    model_orig = joblib.load(f'{model_dir}/random_forest_original.pkl')
    model_pca = joblib.load(f'{model_dir}/random_forest_pca.pkl')
    model_lda = joblib.load(f'{model_dir}/random_forest_lda.pkl')
    pca = joblib.load(f'{model_dir}/pca.pkl')
    lda = joblib.load(f'{model_dir}/lda.pkl')
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    encoder = joblib.load(f'{model_dir}/encoder.pkl')
    numerical_cols = joblib.load(f'{model_dir}/numerical_cols.pkl')
    categorical_cols = joblib.load(f'{model_dir}/categorical_cols.pkl')
    
    return {
        'model_orig': model_orig,
        'model_pca': model_pca,
        'model_lda': model_lda,
        'pca': pca,
        'lda': lda,
        'scaler': scaler,
        'encoder': encoder,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }

if __name__ == "__main__":
    # Entrenar y guardar modelos
    print("Cargando datos...")
    df, _, _ = load_and_preprocess_data(use_sample=True)
    print(f"Datos cargados: {df.shape}")
    
    print("Entrenando modelos...")
    models = train_models_pipeline(df)
    
    print("Guardando modelos...")
    save_models(models)
    
    print("¡Proceso completado!")

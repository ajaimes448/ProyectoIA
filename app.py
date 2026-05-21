# app.py - Sistema de Detección de Fraude Bancario
# TODO EN UN SOLO ARCHIVO

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
st.set_page_config(page_title="Detección de Fraude", page_icon="💰", layout="wide")
st.title("💰 Sistema de Detección de Fraude Bancario")

# Constantes
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ==================== FUNCIONES ====================
def generate_sample_data(n_samples=3000):
    """Genera datos de ejemplo"""
    np.random.seed(RANDOM_STATE)
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
    
    fraud_prob = (
        (df['anomaly_score'] > 0.7) * 0.3 +
        (df['login_attempts'] > 5) * 0.2 +
        (df['suspicious_ip_flag'] == 1) * 0.2 +
        (df['international_transaction_flag'] == 1) * 0.15 +
        (df['transaction_amount'] > 50000) * 0.15
    )
    df['fraud_flag'] = (np.random.random(n_samples) < fraud_prob).astype(int)
    return df

def train_models(df):
    """Entrena los modelos"""
    # Preprocesamiento
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    numerical_cols = [col for col in df_encoded.select_dtypes(include=['int64', 'float64']).columns 
                     if col != 'fraud_flag']
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    # Split
    X = df_encoded.drop(columns=['fraud_flag'])
    y = df_encoded['fraud_flag']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Modelo Original
    model_orig = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=RANDOM_STATE)
    model_orig.fit(x_train, y_train)
    
    # Modelo PCA
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca_train = pca.fit_transform(x_train)
    X_pca_test = pca.transform(x_test)
    model_pca = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=RANDOM_STATE)
    model_pca.fit(X_pca_train, y_train)
    
    return {
        'model_orig': model_orig,
        'model_pca': model_pca,
        'pca': pca,
        'scaler': scaler,
        'encoders': encoders,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'x_test': x_test,
        'y_test': y_test,
        'x_test_pca': X_pca_test,
        'feature_names': x_test.columns.tolist()
    }

def predict_transaction(models, transaction_data):
    """Predicción de una transacción"""
    input_df = pd.DataFrame([transaction_data])
    
    # Codificar categóricas
    for col in models['categorical_cols']:
        if col in input_df.columns and col in models['encoders']:
            input_df[col] = models['encoders'][col].transform(input_df[col].astype(str))
    
    # Escalar numéricas
    input_df[models['numerical_cols']] = models['scaler'].transform(input_df[models['numerical_cols']])
    input_df = input_df[models['feature_names']]
    
    # Predicciones
    pred_orig = models['model_orig'].predict(input_df)[0]
    prob_orig = models['model_orig'].predict_proba(input_df)[0][1]
    
    X_pca = models['pca'].transform(input_df)
    pred_pca = models['model_pca'].predict(X_pca)[0]
    prob_pca = models['model_pca'].predict_proba(X_pca)[0][1]
    
    consensus = 1 if (pred_orig + pred_pca) >= 2 else 0
    
    return {
        'original': {'prediction': int(pred_orig), 'probability': float(prob_orig)},
        'pca': {'prediction': int(pred_pca), 'probability': float(prob_pca)},
        'consensus': consensus
    }

# ==================== CARGA DE DATOS ====================
@st.cache_data
def load_data():
    return generate_sample_data()

@st.cache_resource
def load_models(df):
    return train_models(df)

with st.spinner("Cargando datos y entrenando modelos..."):
    df = load_data()
    models = load_models(df)

# ==================== INTERFAZ ====================
menu = st.sidebar.radio("Menú", ["📊 Dashboard", "🔍 Predicción", "ℹ️ Acerca de"])

if menu == "📊 Dashboard":
    st.header("📊 Dashboard")
    
    col1, col2, col3 = st.columns(3)
    fraud_count = df['fraud_flag'].sum()
    fraud_pct = (fraud_count / len(df)) * 100
    
    col1.metric("Total Transacciones", f"{len(df):,}")
    col2.metric("Fraude Detectado", f"{fraud_count:,}")
    col3.metric("% Fraude", f"{fraud_pct:.2f}%")
    
    st.subheader("📈 Precisión de Modelos")
    acc_orig = accuracy_score(models['y_test'], models['model_orig'].predict(models['x_test']))
    acc_pca = accuracy_score(models['y_test'], models['model_pca'].predict(models['x_test_pca']))
    
    col1, col2 = st.columns(2)
    col1.metric("Random Forest", f"{acc_orig:.2%}")
    col2.metric("Random Forest + PCA", f"{acc_pca:.2%}")
    
    st.subheader("📊 Matriz de Confusión")
    cm = confusion_matrix(models['y_test'], models['model_orig'].predict(models['x_test']))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    st.subheader("📊 Distribución de Fraude")
    fig, ax = plt.subplots()
    df['fraud_flag'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
    ax.set_xticklabels(['No Fraude', 'Fraude'])
    st.pyplot(fig)

elif menu == "🔍 Predicción":
    st.header("🔍 Predicción Individual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Monto", value=10000.0)
        login = st.number_input("Intentos Login", value=3)
        risk = st.slider("Riesgo Dispositivo", 0, 100, 50)
        anomaly = st.slider("Anomaly Score", 0.0, 1.0, 0.3)
    
    with col2:
        international = st.selectbox("Transacción Internacional", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        suspicious_ip = st.selectbox("IP Sospechosa", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        channel = st.selectbox("Canal", ['ATM', 'Mobile App', 'POS Terminal', 'Web Banking'])
        auth = st.selectbox("Autenticación", ['Biometric', 'OTP', 'Password Only'])
    
    if st.button("🔍 Predecir", type="primary"):
        data = {
            'transaction_amount': amount,
            'login_attempts': login,
            'device_risk_score': risk,
            'transfer_frequency': 30,
            'anomaly_score': anomaly,
            'account_age_days': 2000,
            'transaction_time_hour': 12,
            'failed_transactions_last_30d': 5,
            'avg_monthly_balance': 250000,
            'daily_transaction_count': 50,
            'geo_distance_km': 5000,
            'session_duration_minutes': 60,
            'transaction_velocity_score': 50,
            'payment_channel': channel,
            'authentication_type': auth,
            'card_present_flag': 1,
            'international_transaction_flag': international,
            'suspicious_ip_flag': suspicious_ip
        }
        
        result = predict_transaction(models, data)
        
        st.markdown("---")
        if result['consensus'] == 1:
            st.error("🚨 ¡ALERTA! Transacción FRAUDULENTA")
        else:
            st.success("✅ Transacción LEGÍTIMA")
        
        col1, col2 = st.columns(2)
        col1.metric("Random Forest", f"{'FRAUDE' if result['original']['prediction'] == 1 else 'LEGÍTIMO'}")
        col2.metric("Random Forest + PCA", f"{'FRAUDE' if result['pca']['prediction'] == 1 else 'LEGÍTIMO'}")
        
        col1.write(f"Probabilidad: {result['original']['probability']:.2%}")
        col2.write(f"Probabilidad: {result['pca']['probability']:.2%}")

else:
    st.header("ℹ️ Acerca de")
    st.markdown("""
    ### Sistema de Detección de Fraude Bancario
    
    **Modelos:**
    - Random Forest
    - Random Forest + PCA
    
    **Tecnologías:**
    - Streamlit
    - Scikit-learn
    - Pandas, NumPy
    """)
    
    st.info(f"📊 Dataset: {len(df):,} transacciones | {df['fraud_flag'].sum():,} fraudes ({df['fraud_flag'].mean():.1%})")

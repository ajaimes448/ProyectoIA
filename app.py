# app.py - Aplicación de Detección de Fraude Bancario
# VERSIÓN DEFINITIVA Y FUNCIONAL

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Importar funciones del modelo
from model import generate_sample_data, train_models_pipeline, predict_transaction

# Configuración de la página
st.set_page_config(
    page_title="Detección de Fraude Bancario",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Sistema de Detección de Fraude Bancario")
st.markdown("---")

# Sidebar
st.sidebar.title("Navegación")
option = st.sidebar.radio(
    "Selecciona una opción:",
    ["📊 Dashboard", "🔍 Predicción Individual", "ℹ️ Acerca de"]
)

# Cargar datos
@st.cache_data
def load_data():
    st.sidebar.markdown("### Configuración de Datos")
    data_source = st.sidebar.radio(
        "Fuente de datos:",
        ["Usar datos de ejemplo", "Subir archivo CSV"]
    )
    if data_source == "Subir archivo CSV":
        file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
        if file is not None:
            df = pd.read_csv(file)
            st.sidebar.success(f"Datos cargados! Shape: {df.shape}")
            return df
        else:
            st.stop()
    else:
        df = generate_sample_data()
        st.sidebar.info("Usando datos de ejemplo generados automáticamente")
        return df

@st.cache_resource
def get_models(df):
    return train_models_pipeline(df)

# Cargar todo
with st.spinner("Cargando datos y entrenando modelos..."):
    df = load_data()
    models = get_models(df)

# ==================== DASHBOARD ====================
if option == "📊 Dashboard":
    st.header("📊 Dashboard de Detección de Fraude")
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    fraud_count = df['fraud_flag'].sum()
    fraud_pct = (fraud_count / len(df)) * 100
    
    with col1:
        st.metric("Total Transacciones", f"{len(df):,}")
    with col2:
        st.metric("Transacciones Fraudulentas", f"{fraud_count:,}")
    with col3:
        st.metric("% Fraude", f"{fraud_pct:.2f}%")
    with col4:
        st.metric("Modelo Principal", "Random Forest")
    
    st.markdown("---")
    
    # Evaluación
    st.subheader("📈 Evaluación de Modelos")
    
    acc_orig = accuracy_score(models['y_test'], models['model'].predict(models['x_test']))
    acc_pca = accuracy_score(models['y_test_pca'], models['model_pca'].predict(models['x_test_pca']))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Modelo Original", f"{acc_orig:.4f}")
    with col2:
        st.metric("Modelo PCA", f"{acc_pca:.4f}")
    
    if models['model_lda'] is not None:
        acc_lda = accuracy_score(models['y_test_lda'], models['model_lda'].predict(models['x_test_lda']))
        st.metric("Modelo LDA", f"{acc_lda:.4f}")
    
    # Matrices de confusión
    st.subheader("📊 Matrices de Confusión")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Modelo Original**")
        cm = confusion_matrix(models['y_test'], models['model'].predict(models['x_test']))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.write("**Modelo PCA**")
        cm = confusion_matrix(models['y_test_pca'], models['model_pca'].predict(models['x_test_pca']))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        plt.close()
    
    # Distribución
    st.subheader("📊 Distribución de Fraude")
    fig, ax = plt.subplots(figsize=(6, 4))
    df['fraud_flag'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
    ax.set_title('Transacciones Fraudulentas vs No Fraudulentas')
    ax.set_xlabel('Fraude')
    ax.set_ylabel('Cantidad')
    ax.set_xticklabels(['No Fraude', 'Fraude'])
    st.pyplot(fig)
    plt.close()

# ==================== PREDICCIÓN INDIVIDUAL ====================
elif option == "🔍 Predicción Individual":
    st.header("🔍 Predicción de Fraude Individual")
    st.markdown("Ingresa los datos de la transacción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_amount = st.number_input("Monto", min_value=0.0, value=10000.0)
        login_attempts = st.number_input("Intentos de Login", min_value=1, max_value=20, value=3)
        device_risk_score = st.slider("Riesgo Dispositivo", 0.0, 100.0, 50.0)
        transfer_frequency = st.number_input("Frecuencia Transferencias", min_value=0, value=30)
        anomaly_score = st.slider("Anomaly Score", 0.0, 1.0, 0.3)
    
    with col2:
        account_age_days = st.number_input("Edad Cuenta (días)", min_value=0, value=2000)
        transaction_time_hour = st.slider("Hora", 0, 23, 12)
        failed_transactions = st.number_input("Fallidas 30d", min_value=0, value=5)
        avg_monthly_balance = st.number_input("Balance Promedio", min_value=0.0, value=250000.0)
        daily_transaction_count = st.number_input("Transacciones/Día", min_value=1, value=50)
    
    col3, col4 = st.columns(2)
    
    with col3:
        geo_distance_km = st.number_input("Distancia (km)", min_value=0, value=5000)
        session_duration_minutes = st.number_input("Duración Sesión (min)", min_value=1, value=60)
        transaction_velocity_score = st.slider("Velocidad Transacción", 0.0, 100.0, 50.0)
    
    with col4:
        payment_channel = st.selectbox("Canal", ['ATM', 'Mobile App', 'POS Terminal', 'Web Banking'])
        authentication_type = st.selectbox("Autenticación", ['Biometric', 'OTP', 'Password Only'])
        card_present_flag = st.selectbox("Tarjeta Presente", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        international_flag = st.selectbox("Internacional", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        suspicious_ip_flag = st.selectbox("IP Sospechosa", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    
    if st.button("🔍 Predecir", type="primary", use_container_width=True):
        transaction_data = {
            'transaction_amount': transaction_amount,
            'login_attempts': login_attempts,
            'device_risk_score': device_risk_score,
            'transfer_frequency': transfer_frequency,
            'anomaly_score': anomaly_score,
            'account_age_days': account_age_days,
            'transaction_time_hour': transaction_time_hour,
            'failed_transactions_last_30d': failed_transactions,
            'avg_monthly_balance': avg_monthly_balance,
            'daily_transaction_count': daily_transaction_count,
            'geo_distance_km': geo_distance_km,
            'session_duration_minutes': session_duration_minutes,
            'transaction_velocity_score': transaction_velocity_score,
            'payment_channel': payment_channel,
            'authentication_type': authentication_type,
            'card_present_flag': card_present_flag,
            'international_transaction_flag': international_flag,
            'suspicious_ip_flag': suspicious_ip_flag
        }
        
        prediction = predict_transaction(models, transaction_data)
        
        st.markdown("---")
        st.subheader("📋 Resultados")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            color = "🔴" if prediction['original']['prediction'] == 1 else "🟢"
            st.metric("Modelo Original", 
                     f"{color} {'FRAUDE' if prediction['original']['prediction'] == 1 else 'NO FRAUDE'}",
                     f"Prob: {prediction['original']['probability']:.2%}")
        
        with col_r2:
            color = "🔴" if prediction['pca']['prediction'] == 1 else "🟢"
            st.metric("Modelo PCA",
                     f"{color} {'FRAUDE' if prediction['pca']['prediction'] == 1 else 'NO FRAUDE'}",
                     f"Prob: {prediction['pca']['probability']:.2%}")
        
        with col_r3:
            color = "🔴" if prediction['lda']['prediction'] == 1 else "🟢"
            st.metric("Modelo LDA",
                     f"{color} {'FRAUDE' if prediction['lda']['prediction'] == 1 else 'NO FRAUDE'}",
                     f"Prob: {prediction['lda']['probability']:.2%}")
        
        st.markdown("---")
        consensus_color = "🔴" if prediction['consensus'] == 1 else "🟢"
        st.markdown(f"### {consensus_color} **Consenso: {'FRAUDE' if prediction['consensus'] == 1 else 'NO FRAUDE'}**")

# ==================== ACERCA DE ====================
else:
    st.header("ℹ️ Acerca del Sistema")
    st.markdown("""
    ### Sistema de Detección de Fraude Bancario
    
    **Modelos implementados:**
    - Random Forest Original
    - Random Forest con PCA
    - Random Forest con LDA
    
    **Tecnologías:** Streamlit, Scikit-learn, Pandas, NumPy
    """)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Registros", f"{len(df):,}")
    with col2:
        st.metric("Tasa de Fraude", f"{(df['fraud_flag'].sum()/len(df)*100):.2f}%")

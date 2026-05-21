import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Importar funciones del modelo
from model import generate_sample_data, train_models_pipeline, predict_transaction

# Configuración de la página
st.set_page_config(
    page_title="Detección de Fraude Bancario",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("💰 Sistema de Detección de Fraude Bancario")
st.markdown("---")

# Sidebar
st.sidebar.title("Navegación")
option = st.sidebar.radio(
    "Selecciona una opción:",
    ["📊 Dashboard", "🔍 Predicción Individual", "📈 Visualizaciones", "ℹ️ Acerca de"]
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

# Cargar datos y modelos
with st.spinner("Cargando datos y entrenando modelos..."):
    df = load_data()
    models = get_models(df)

# ==================== DASHBOARD ====================
if option == "📊 Dashboard":
    st.header("📊 Dashboard de Detección de Fraude")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transacciones", f"{len(df):,}")
    with col2:
        fraud_count = df['fraud_flag'].sum()
        st.metric("Transacciones Fraudulentas", f"{fraud_count:,}")
    with col3:
        fraud_pct = (fraud_count / len(df)) * 100 if len(df) > 0 else 0
        st.metric("% Fraude", f"{fraud_pct:.2f}%")
    with col4:
        st.metric("Modelo Principal", "Random Forest")
    
    st.markdown("---")
    
    # Evaluación de modelos
    st.subheader("📈 Evaluación de Modelos")
    
    acc_orig = accuracy_score(models['y_test'], models['model'].predict(models['x_test']))
    acc_pca = accuracy_score(models['y_test_pca'], models['model_pca'].predict(models['x_test_pca']))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modelo Original", f"{acc_orig:.4f}")
    with col2:
        st.metric("Modelo PCA", f"{acc_pca:.4f}")
    
    if models.get('model_lda') is not None:
        acc_lda = accuracy_score(models['y_test_lda'], models['model_lda'].predict(models['x_test_lda']))
        with col3:
            st.metric("Modelo LDA", f"{acc_lda:.4f}")
        best_model = max([("Original", acc_orig), ("PCA", acc_pca), ("LDA", acc_lda)], key=lambda x: x[1])
    else:
        best_model = max([("Original", acc_orig), ("PCA", acc_pca)], key=lambda x: x[1])
    
    st.success(f"🏆 Mejor modelo: **{best_model[0]}** con accuracy de **{best_model[1]:.4f}**")
    
    # Matrices de confusión
    st.subheader("📊 Matrices de Confusión")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Modelo Original**")
        cm_orig = confusion_matrix(models['y_test'], models['model'].predict(models['x_test']))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.write("**Modelo PCA**")
        cm_pca = confusion_matrix(models['y_test_pca'], models['model_pca'].predict(models['x_test_pca']))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Greens', ax=ax)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        plt.close()
    
    # Distribución de fraude
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
    st.markdown("Ingresa los datos de la transacción para predecir si es fraudulenta o no.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_amount = st.number_input("Monto de Transacción", min_value=0.0, value=10000.0, step=500.0)
        login_attempts = st.number_input("Intentos de Login", min_value=1, max_value=20, value=3)
        device_risk_score = st.slider("Puntaje de Riesgo del Dispositivo", 0.0, 100.0, 50.0)
        transfer_frequency = st.number_input("Frecuencia de Transferencias", min_value=0, max_value=100, value=30)
        anomaly_score = st.slider("Puntaje de Anomalía", 0.0, 1.0, 0.3, step=0.01)
    
    with col2:
        account_age_days = st.number_input("Edad de la Cuenta (días)", min_value=0, max_value=5000, value=2000)
        transaction_time_hour = st.slider("Hora de Transacción", 0, 23, 12)
        failed_transactions = st.number_input("Transacciones Fallidas (últimos 30 días)", min_value=0, max_value=50, value=5)
        avg_monthly_balance = st.number_input("Balance Promedio Mensual", min_value=0.0, value=250000.0, step=10000.0)
        daily_transaction_count = st.number_input("Transacciones Diarias", min_value=1, max_value=200, value=50)
    
    col3, col4 = st.columns(2)
    
    with col3:
        geo_distance_km = st.number_input("Distancia Geográfica (km)", min_value=0, max_value=20000, value=5000)
        session_duration_minutes = st.number_input("Duración de Sesión (minutos)", min_value=1, max_value=300, value=60)
        transaction_velocity_score = st.slider("Puntaje de Velocidad de Transacción", 0.0, 100.0, 50.0)
    
    with col4:
        payment_channel = st.selectbox("Canal de Pago", ['ATM', 'Mobile App', 'POS Terminal', 'Web Banking'])
        authentication_type = st.selectbox("Tipo de Autenticación", ['Biometric', 'OTP', 'Password Only'])
        card_present_flag = st.selectbox("Tarjeta Presente", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        international_flag = st.selectbox("Transacción Internacional", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        suspicious_ip_flag = st.selectbox("IP Sospechosa", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    
    # Botón de predicción
    if st.button("🔍 Predecir Fraude", type="primary", use_container_width=True):
        # Crear diccionario con los datos
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
        
        # Realizar predicción
        prediction = predict_transaction(models, transaction_data)
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("📋 Resultados de la Predicción")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            color = "🔴" if prediction['original']['prediction'] == 1 else "🟢"
            st.metric("Modelo Original", f"{color} {'FRAUDE' if prediction['original']['prediction'] == 1 else 'NO FRAUDE'}", 
                     f"Probabilidad: {prediction['original']['probability']:.2%}")
        
        with col_r2:
            color = "🔴" if prediction['pca']['prediction'] == 1 else "🟢"
            st.metric("Modelo PCA", f"{color} {'FRAUDE' if prediction['pca']['prediction'] == 1 else 'NO FRAUDE'}",
                     f"Probabilidad: {prediction['pca']['probability']:.2%}")
        
        with col_r3:
            color = "🔴" if prediction['lda']['prediction'] == 1 else "🟢"
            st.metric("Modelo LDA", f"{color} {'FRAUDE' if prediction['lda']['prediction'] == 1 else 'NO FRAUDE'}",
                     f"Probabilidad: {prediction['lda']['probability']:.2%}")
        
        # Consenso
        st.markdown("---")
        consensus_color = "🔴" if prediction['consensus'] == 1 else "🟢"
        st.markdown(f"### {consensus_color} **Consenso: {'TRANSACCIÓN FRAUDULENTA' if prediction['consensus'] == 1 else 'TRANSACCIÓN LEGÍTIMA'}**")
        
        if prediction['consensus'] == 1:
            st.warning("⚠️ La transacción ha sido identificada como potencialmente fraudulenta. Se recomienda verificación adicional.")
        else:
            st.success("✅ La transacción parece legítima según los modelos.")

# ==================== VISUALIZACIONES ====================
elif option == "📈 Visualizaciones":
    st.header("📈 Visualizaciones de Datos")
    
    st.markdown("### Distribución de Variables Numéricas")
    
    numerical_vars = ['transaction_amount', 'login_attempts', 'device_risk_score', 
                      'anomaly_score', 'account_age_days', 'daily_transaction_count']
    
    numerical_vars = [var for var in numerical_vars if var in df.columns]
    
    cols = st.columns(2)
    
    for i, var in enumerate(numerical_vars):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[df['fraud_flag'] == 0][var].hist(alpha=0.7, label='No Fraude', bins=30, ax=ax)
            df[df['fraud_flag'] == 1][var].hist(alpha=0.7, label='Fraude', bins=30, ax=ax)
            ax.set_xlabel(var)
            ax.set_ylabel('Frecuencia')
            ax.legend()
            ax.set_title(f'Distribución de {var}')
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    # Matriz de correlación
    st.subheader("📊 Matriz de Correlación")
    
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
        ax.set_title('Matriz de Correlación de Variables Numéricas')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No hay suficientes variables numéricas para mostrar la matriz de correlación.")

# ==================== ACERCA DE ====================
else:
    st.header("ℹ️ Acerca del Sistema")
    
    st.markdown("""
    ### Sistema de Detección de Fraude Bancario
    
    Esta aplicación utiliza técnicas de **Machine Learning** para identificar transacciones bancarias potencialmente fraudulentas.
    
    #### Modelos Implementados:
    
    - **Random Forest Original**: Clasificador basado en todas las características originales
    - **Random Forest con PCA**: Reducción de dimensionalidad mediante Análisis de Componentes Principales
    - **Random Forest con LDA**: Análisis Discriminante Lineal para maximizar separación entre clases
    
    #### Tecnologías Utilizadas:
    
    - **Frontend**: Streamlit
    - **Machine Learning**: Scikit-learn
    - **Visualización**: Matplotlib, Seaborn
    - **Procesamiento de Datos**: Pandas, NumPy
    
    #### Características:
    
    - Dashboard interactivo con métricas en tiempo real
    - Predicción individual de transacciones
    - Visualizaciones exploratorias de datos
    - Soporte para carga de datasets personalizados
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Estadísticas del Dataset Actual")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Registros", f"{len(df):,}")
        st.metric("Características", f"{len(df.columns) - 1}")
    with col2:
        st.metric("Transacciones Fraudulentas", f"{df['fraud_flag'].sum():,}")
        st.metric("Tasa de Fraude", f"{(df['fraud_flag'].sum()/len(df)*100):.2f}%" if len(df) > 0 else "0.00%")

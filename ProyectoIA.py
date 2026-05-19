# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io
import os

# Configurar página
st.set_page_config(
    page_title="Detección de Fraude Bancario",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("💰 Sistema de Detección de Fraude Bancario")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("Navegación")
option = st.sidebar.radio(
    "Selecciona una opción:",
    ["📊 Dashboard", "🔍 Predicción Individual", "📈 Visualizaciones", "ℹ️ Acerca de"]
)

# Función para generar datos de ejemplo
def generate_sample_data():
    """Genera datos de ejemplo para demostración"""
    np.random.seed(42)
    n_samples = 10000
    
    # Generar características
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
    
    # Generar etiqueta de fraude basada en reglas
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
    
    return df

# Función para cargar y preprocesar datos
@st.cache_data
def load_data():
    """Carga y preprocesa el dataset"""
    st.sidebar.markdown("### Configuración de Datos")
    data_source = st.sidebar.radio(
        "Fuente de datos:",
        ["Usar datos de ejemplo", "Subir archivo CSV"]
    )
    
    df = None
    
    if data_source == "Subir archivo CSV":
        file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
        if file is not None:
            df = pd.read_csv(file)
            st.sidebar.success(f"Datos cargados! Shape: {df.shape}")
    else:
        df = generate_sample_data()
        st.sidebar.info("Usando datos de ejemplo generados automáticamente")
    
    if df is None:
        st.error("No se pudieron cargar los datos. Por favor, sube un archivo CSV o usa los datos de ejemplo.")
        st.stop()
    
    return df

@st.cache_resource
def train_models(df):
    """Entrena los modelos necesarios"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Copiar datos
    df_processed = df.copy()
    
    # Codificar variables categóricas de manera individual
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        encoders[col] = le
    
    # Escalar variables numéricas
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'fraud_flag']
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    # Preparar X e y
    X = df_processed.drop(columns=['fraud_flag'])
    y = df_processed['fraud_flag']
    
    # Dividir datos
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Entrenar modelo Random Forest original
    model = RandomForestClassifier(
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
    model.fit(x_train, y_train)
    
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
    
    # LDA (requiere al menos 2 clases)
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
        'model': model,
        'model_pca': model_pca,
        'model_lda': model_lda,
        'pca': pca,
        'lda': lda,
        'scaler': scaler,
        'encoders': encoders,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'x_test': x_test,
        'y_test': y_test,
        'x_test_pca': X_pca_test_combined,
        'y_test_pca': y_test,
        'x_test_lda': X_lda_test_combined if model_lda else None,
        'y_test_lda': y_test if model_lda else None
    }

# Cargar datos y modelos
with st.spinner("Cargando datos y entrenando modelos..."):
    df = load_data()
    models = train_models(df)

# ==================== DASHBOARD ====================
if option == "📊 Dashboard":
    st.header("📊 Dashboard de Detección de Fraude")
    
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
    
    if models['model_lda'] is not None:
        acc_lda = accuracy_score(models['y_test_lda'], models['model_lda'].predict(models['x_test_lda']))
        with col3:
            st.metric("Modelo LDA", f"{acc_lda:.4f}")
        
        # Mejor modelo
        best_model = max([("Original", acc_orig), ("PCA", acc_pca), ("LDA", acc_lda)], key=lambda x: x[1])
    else:
        best_model = max([("Original", acc_orig), ("PCA", acc_pca)], key=lambda x: x[1])
    
    st.success(f"🏆 Mejor modelo: **{best_model[0]}** con accuracy de **{best_model[1]:.4f}**")
    
    # Reporte de clasificación
    with st.expander("📋 Ver Reporte de Clasificación Detallado"):
        st.subheader("Modelo Original")
        y_pred_orig = models['model'].predict(models['x_test'])
        report_orig = classification_report(models['y_test'], y_pred_orig, output_dict=True)
        st.dataframe(pd.DataFrame(report_orig).transpose())
        
        st.subheader("Modelo PCA")
        y_pred_pca = models['model_pca'].predict(models['x_test_pca'])
        report_pca = classification_report(models['y_test_pca'], y_pred_pca, output_dict=True)
        st.dataframe(pd.DataFrame(report_pca).transpose())
        
        if models['model_lda'] is not None:
            st.subheader("Modelo LDA")
            y_pred_lda = models['model_lda'].predict(models['x_test_lda'])
            report_lda = classification_report(models['y_test_lda'], y_pred_lda, output_dict=True)
            st.dataframe(pd.DataFrame(report_lda).transpose())
    
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
    
    if models['model_lda'] is not None:
        st.subheader("Modelo LDA")
        cm_lda = confusion_matrix(models['y_test_lda'], models['model_lda'].predict(models['x_test_lda']))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Reds', ax=ax)
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
    
    # Anomaly Score Distribution
    st.subheader("📊 Distribución de Anomaly Score")
    fig, ax = plt.subplots(figsize=(10, 5))
    df[df['fraud_flag'] == 0]['anomaly_score'].hist(alpha=0.7, label='No Fraude', bins=30, ax=ax)
    df[df['fraud_flag'] == 1]['anomaly_score'].hist(alpha=0.7, label='Fraude', bins=30, ax=ax)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frecuencia')
    ax.legend()
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
        input_data = {
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
        
        # Convertir a DataFrame y preprocesar
        input_df = pd.DataFrame([input_data])
        
        # Codificar variables categóricas
        for col in models['categorical_cols']:
            if col in input_df.columns:
                if col in models['encoders']:
                    try:
                        input_df[col] = models['encoders'][col].transform(input_df[col].astype(str))
                    except ValueError:
                        input_df[col] = 0
        
        # CORRECCIÓN AQUÍ: Escalar todas las variables numéricas juntas
        input_df[models['numerical_cols']] = models['scaler'].transform(input_df[models['numerical_cols']])
        
        # Reordenar las columnas del dataframe para que coincidan exactamente con x_test
        input_df = input_df[models['x_test'].columns]

        # Predicción con el modelo original
        pred_orig = models['model'].predict(input_df)[0]
        prob_orig = models['model'].predict_proba(input_df)[0][1]
        
        # PCA
        X_pca = models['pca'].transform(input_df[models['numerical_cols']])
        if len(models['categorical_cols']) > 0:
            X_pca_combined = np.concatenate([X_pca, input_df[models['categorical_cols']].values], axis=1)
        else:
            X_pca_combined = X_pca
        pred_pca = models['model_pca'].predict(X_pca_combined)[0]
        prob_pca = models['model_pca'].predict_proba(X_pca_combined)[0][1]
        
        # LDA
        if models['model_lda'] is not None and models['lda'] is not None:
            X_lda = models['lda'].transform(input_df[models['numerical_cols']])
            if len(models['categorical_cols']) > 0:
                X_lda_combined = np.concatenate([X_lda, input_df[models['categorical_cols']].values], axis=1)
            else:
                X_lda_combined = X_lda
            pred_lda = models['model_lda'].predict(X_lda_combined)[0]
            prob_lda = models['model_lda'].predict_proba(X_lda_combined)[0][1]
        else:
            pred_lda = pred_orig
            prob_lda = prob_orig
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("📋 Resultados de la Predicción")
        
        # Determinar el consenso
        predictions = [pred_orig, pred_pca, pred_lda]
        consensus = max(set(predictions), key=predictions.count)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            color = "🔴" if pred_orig == 1 else "🟢"
            st.metric("Modelo Original", f"{color} {'FRAUDE' if pred_orig == 1 else 'NO FRAUDE'}", 
                     f"Probabilidad: {prob_orig:.2%}")
        
        with col_r2:
            color = "🔴" if pred_pca == 1 else "🟢"
            st.metric("Modelo PCA", f"{color} {'FRAUDE' if pred_pca == 1 else 'NO FRAUDE'}",
                     f"Probabilidad: {prob_pca:.2%}")
        
        with col_r3:
            color = "🔴" if pred_lda == 1 else "🟢"
            st.metric("Modelo LDA", f"{color} {'FRAUDE' if pred_lda == 1 else 'NO FRAUDE'}",
                     f"Probabilidad: {prob_lda:.2%}")
        
        # Consenso y recomendación
        st.markdown("---")
        consensus_color = "🔴" if consensus == 1 else "🟢"
        st.markdown(f"### {consensus_color} **Consenso: {'TRANSACCIÓN FRAUDULENTA' if consensus == 1 else 'TRANSACCIÓN LEGÍTIMA'}**")
        
        if consensus == 1:
            st.warning("⚠️ La transacción ha sido identificada como potencialmente fraudulenta. Se recomienda verificación adicional.")
        else:
            st.success("✅ La transacción parece legítima según los modelos.")
        
        # Mostrar factores de riesgo si es fraude
        if consensus == 1:
            st.markdown("---")
            st.subheader("🔍 Factores de Riesgo Detectados")
            risks = []
            if anomaly_score > 0.7:
                risks.append("• Puntaje de anomalía elevado")
            if login_attempts > 8:
                risks.append("• Múltiples intentos de login")
            if suspicious_ip_flag == 1:
                risks.append("• IP sospechosa detectada")
            if international_flag == 1 and geo_distance_km > 5000:
                risks.append("• Transacción internacional con gran distancia geográfica")
            if transaction_amount > 20000:
                risks.append("• Monto de transacción elevado")
            
            for risk in risks:
                st.write(risk)
            
            if not risks:
                st.write("No se identificaron factores de riesgo específicos.")

# ==================== VISUALIZACIONES ====================
elif option == "📈 Visualizaciones":
    st.header("📈 Visualizaciones de Datos")
    
    st.markdown("### Distribución de Variables Numéricas")
    
    numerical_vars = ['transaction_amount', 'login_attempts', 'device_risk_score', 
                      'anomaly_score', 'account_age_days', 'daily_transaction_count']
    
    # Filtrar variables que existen en el dataframe
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
    
    # Gráfico de correlación
    st.subheader("📊 Matriz de Correlación")
    
    # Seleccionar solo columnas numéricas para correlación
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
        ax.set_title('Matriz de Correlación de Variables Numéricas')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No hay suficientes variables numéricas para mostrar la matriz de correlación.")
    
    st.markdown("---")
    
    # Boxplots por tipo de fraude
    st.subheader("📊 Boxplots por Tipo de Fraude")
    
    selected_var = st.selectbox("Selecciona una variable:", numerical_vars)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column=selected_var, by='fraud_flag', ax=ax)
    ax.set_title(f'Boxplot de {selected_var} por Tipo de Fraude')
    ax.set_xlabel('Fraude')
    ax.set_xticklabels(['No Fraude', 'Fraude'])
    st.pyplot(fig)
    plt.close()

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
    
    #### Características del Dataset de Ejemplo:
    
    - **Total de transacciones**: 10,000
    - **Variables**: 18 características (numéricas y categóricas)
    - **Desequilibrio de clases**: Ajustado automáticamente
    
    #### Rendimiento del Mejor Modelo:
    
    """)
    
    # Mostrar métricas del mejor modelo
    acc_orig = accuracy_score(models['y_test'], models['model'].predict(models['x_test']))
    acc_pca = accuracy_score(models['y_test_pca'], models['model_pca'].predict(models['x_test_pca']))
    
    best_models = [("Original", acc_orig), ("PCA", acc_pca)]
    
    if models['model_lda'] is not None:
        acc_lda = accuracy_score(models['y_test_lda'], models['model_lda'].predict(models['x_test_lda']))
        best_models.append(("LDA", acc_lda))
    
    best_model = max(best_models, key=lambda x: x[1])
    
    st.info(f"""
    - **Mejor modelo**: {best_model[0]}
    - **Accuracy**: {best_model[1]:.4f}
    - **Precisión en detección de fraude**: Modelo especializado para detectar transacciones fraudulentas
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

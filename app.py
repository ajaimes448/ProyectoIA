# app.py - Sistema de Detección de Fraude Bancario
# VERSIÓN CORREGIDA - Solo texto visible

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Estilo CORREGIDO - Texto visible
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.2rem;
    }
    .metric-card h3 {
        font-size: 0.9rem;
        margin: 0;
        color: #000000 !important;
    }
    .metric-card h2 {
        font-size: 1.5rem;
        margin: 0;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Inicialización
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.df = None
    st.session_state.models = None
    st.session_state.step = 1

st.title("Sistema de Detección de Fraude Bancario")
st.markdown("---")

# ==================== PASO 1: CARGA DE DATOS ====================
st.header("Paso 1: Carga de Datos")

if st.session_state.df is None:
    
    data_source = st.radio(
        "Fuente de datos:",
        ["Datos de Ejemplo", "Subir CSV", "Kaggle Dataset"],
        horizontal=True
    )
    
    if data_source == "Datos de Ejemplo":
        if st.button("Generar datos", use_container_width=True):
            with st.spinner("Generando datos..."):
                n_samples = 3000
                np.random.seed(42)
                
                df = pd.DataFrame({
                    'transaction_amount': np.random.exponential(10000, n_samples),
                    'login_attempts': np.random.poisson(2, n_samples),
                    'device_risk_score': np.random.uniform(0, 100, n_samples),
                    'transfer_frequency': np.random.poisson(30, n_samples),
                    'anomaly_score': np.random.beta(2, 5, n_samples),
                    'account_age_days': np.random.exponential(1000, n_samples),
                    'transaction_time_hour': np.random.randint(0, 24, n_samples),
                    'failed_transactions': np.random.poisson(2, n_samples),
                    'avg_monthly_balance': np.random.exponential(500000, n_samples),
                    'daily_transaction_count': np.random.poisson(50, n_samples),
                    'geo_distance_km': np.random.exponential(1000, n_samples),
                    'session_duration': np.random.exponential(30, n_samples),
                    'payment_channel': np.random.choice(['ATM', 'Mobile', 'POS', 'Web'], n_samples),
                    'international': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                    'suspicious_ip': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
                })
                
                fraud_prob = (
                    (df['anomaly_score'] > 0.7) * 0.3 +
                    (df['login_attempts'] > 5) * 0.2 +
                    (df['suspicious_ip'] == 1) * 0.2 +
                    (df['international'] == 1) * 0.15 +
                    (df['transaction_amount'] > 50000) * 0.15
                )
                df['fraud_flag'] = (np.random.random(n_samples) < fraud_prob).astype(int)
                
                st.session_state.df = df
                st.session_state.step = 2
                st.rerun()
    
    elif data_source == "Subir CSV":
        uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.step = 2
            st.rerun()
    
    else:
        st.info("Para Kaggle: pega el path del dataset")
        kaggle_path = st.text_input("Path:", placeholder="mlg-od/credit-card-fraud")
        if st.button("Cargar") and kaggle_path:
            try:
                import kagglehub
                path = kagglehub.dataset_download(kaggle_path)
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.csv'):
                            df = pd.read_csv(os.path.join(root, file))
                            st.session_state.df = df
                            st.session_state.step = 2
                            st.rerun()
                            break
                    break
            except Exception as e:
                st.error(f"Error: {e}")

# ==================== PASO 2: CONFIGURACIÓN ====================
if st.session_state.df is not None and st.session_state.step == 2 and not st.session_state.model_trained:
    st.header("Paso 2: Configuración del Modelo")
    
    df = st.session_state.df
    
    with st.expander("Vista previa"):
        st.dataframe(df.head(10))
    
    target_column = st.selectbox("Columna objetivo:", df.columns)
    
    # Mostrar información básica
    if df[target_column].dtype in ['int64', 'float64']:
        fraud_count = df[target_column].sum()
        total = len(df)
        st.write(f"Casos positivos: {fraud_count} ({fraud_count/total:.1%})")
    
    all_cols = [c for c in df.columns if c != target_column]
    numeric_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[all_cols].select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        selected_numeric = st.multiselect("Variables numéricas:", numeric_cols, default=numeric_cols[:3])
    with col2:
        selected_categorical = st.multiselect("Variables categóricas:", categorical_cols, default=categorical_cols[:1])
    
    selected_features = selected_numeric + selected_categorical
    
    if not selected_features:
        st.warning("Selecciona al menos una característica")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Tamaño de prueba:", 0.1, 0.4, 0.2, 0.05)
    with col2:
        model_type = st.selectbox("Algoritmo:", ["Random Forest", "Logistic Regression", "Decision Tree"])
    with col3:
        use_balance = st.checkbox("Balancear clases")
    
    if st.button("Entrenar Modelo", type="primary", use_container_width=True):
        with st.spinner("Entrenando..."):
            try:
                X = df[selected_features].copy()
                y = df[target_column].copy()
                
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                encoders = {}
                for col in selected_categorical:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le
                
                scaler = StandardScaler()
                if selected_numeric:
                    X[selected_numeric] = scaler.fit_transform(X[selected_numeric])
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() <= 10 else None
                )
                
                if model_type == "Random Forest":
                    if use_balance:
                        model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
                    else:
                        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                elif model_type == "Logistic Regression":
                    if use_balance:
                        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
                    else:
                        model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    if use_balance:
                        model = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
                    else:
                        model = DecisionTreeClassifier(max_depth=10, random_state=42)
                
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.0
                
                st.session_state.models = {
                    'model': model,
                    'scaler': scaler,
                    'encoders': encoders,
                    'selected_features': selected_features,
                    'selected_numeric': selected_numeric,
                    'selected_categorical': selected_categorical,
                    'target_column': target_column,
                    'model_type': model_type,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                st.session_state.model_trained = True
                st.session_state.step = 3
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== PASO 3: RESULTADOS ====================
if st.session_state.model_trained and st.session_state.step == 3:
    st.header("Paso 3: Resultados")
    
    models = st.session_state.models
    
    # Métricas con tarjetas de texto visible
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>ACCURACY</h3>
            <h2>{models['accuracy']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>PRECISION</h3>
            <h2>{models['precision']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>RECALL</h3>
            <h2>{models['recall']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>F1 SCORE</h3>
            <h2>{models['f1']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h3>ROC AUC</h3>
            <h2>{models['auc']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Diagnóstico
    if models['recall'] < 0.5:
        st.warning("El recall es bajo. El modelo no detecta suficientes fraudes. Activa 'Balancear clases' para mejorar.")
    
    # Matriz de confusión
    st.subheader("Matriz de Confusión")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(models['y_test'], models['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    st.pyplot(fig)
    plt.close()
    
    # Botones
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reiniciar", use_container_width=True):
            for key in ['df', 'model_trained', 'models', 'step']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col2:
        if st.button("Hacer Predicciones", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()

# ==================== PASO 4: PREDICCIONES ====================
if st.session_state.model_trained and st.session_state.step == 4:
    st.header("Paso 4: Predicción")
    
    models = st.session_state.models
    df = st.session_state.df
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        input_data = {}
        
        with col1:
            st.markdown("**Variables Numéricas**")
            for feature in models['selected_numeric']:
                default_val = float(df[feature].mean()) if feature in df.columns else 0.0
                input_data[feature] = st.number_input(f"{feature}:", value=default_val, key=f"num_{feature}")
        
        with col2:
            if models['selected_categorical']:
                st.markdown("**Variables Categóricas**")
                for feature in models['selected_categorical']:
                    if feature in df.columns:
                        unique_values = df[feature].dropna().unique().tolist()
                        input_data[feature] = st.selectbox(f"{feature}:", unique_values, key=f"cat_{feature}")
        
        submitted = st.form_submit_button("Predecir", use_container_width=True, type="primary")
        
        if submitted:
            try:
                input_df = pd.DataFrame([input_data])
                
                for col in models['selected_categorical']:
                    if col in models['encoders']:
                        input_df[col] = models['encoders'][col].transform(input_df[col].astype(str))
                
                if models['selected_numeric']:
                    input_df[models['selected_numeric']] = models['scaler'].transform(input_df[models['selected_numeric']])
                
                input_df = input_df[models['selected_features']]
                prediction = models['model'].predict(input_df)[0]
                
                if prediction == 1:
                    st.error("PREDICCION: FRAUDE")
                else:
                    st.success("PREDICCION: LEGITIMO")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.button("Volver"):
        st.session_state.step = 3
        st.rerun()

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray'>Sistema de Detección de Fraude Bancario</p>", unsafe_allow_html=True)

# app.py - Sistema de Detección de Fraude Bancario
# VERSIÓN CORREGIDA - Texto visible en todos los cuadros

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

# Estilo CORREGIDO - Texto oscuro sobre fondo claro
st.markdown("""
<style>
    /* Estilo para tarjetas de métricas - TEXTO OSCURO */
    .metric-card {
        background-color: #ffffff;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.2rem;
        border: 1px solid #ddd;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        font-size: 0.9rem;
        margin: 0 0 0.5rem 0;
        color: #333333 !important;
        font-weight: 600;
    }
    .metric-card h2 {
        font-size: 1.8rem;
        margin: 0;
        font-weight: bold;
    }
    .metric-card p {
        font-size: 0.75rem;
        margin: 0.3rem 0 0 0;
        color: #666666;
    }
    
    /* Color para textos dentro de st.info, st.success, etc */
    .stAlert {
        color: #000000 !important;
    }
    .stAlert div {
        color: #000000 !important;
    }
    
    /* Tablas y dataframes */
    .dataframe {
        color: #000000 !important;
    }
    
    /* Inputs y textos en general */
    .stMarkdown, .stText, label, .stSelectbox label, .stMultiSelect label {
        color: #000000 !important;
    }
    
    /* Números en los sliders */
    .stSlider label, .stSlider div {
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
                n_samples = 5000
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
                    'international': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
                    'suspicious_ip': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
                })
                
                fraud_prob = np.zeros(n_samples)
                fraud_prob += (df['anomaly_score'] > 0.6) * 0.3
                fraud_prob += (df['login_attempts'] > 4) * 0.25
                fraud_prob += (df['suspicious_ip'] == 1) * 0.25
                fraud_prob += (df['international'] == 1) * 0.15
                fraud_prob += (df['transaction_amount'] > 40000) * 0.15
                fraud_prob += (df['device_risk_score'] > 80) * 0.2
                
                df['fraud_flag'] = (np.random.random(n_samples) < fraud_prob).astype(int)
                
                if df['fraud_flag'].sum() < n_samples * 0.05:
                    extra_frauds = int(n_samples * 0.05) - df['fraud_flag'].sum()
                    fraud_indices = np.random.choice(df[df['fraud_flag'] == 0].index, extra_frauds, replace=False)
                    df.loc[fraud_indices, 'fraud_flag'] = 1
                
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
    
    with st.expander("Vista previa de los datos"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Selección de columna objetivo
    st.subheader("Columna objetivo")
    
    target_column = st.selectbox(
        "Selecciona la columna a predecir:",
        df.columns
    )
    
    # Mostrar distribución
    if df[target_column].dtype in ['int64', 'float64']:
        fraud_count = df[target_column].sum()
        total = len(df)
        st.write(f"**Distribución:** {fraud_count} casos positivos ({fraud_count/total:.1%})")
        st.write(f"**Valores únicos:** {df[target_column].nunique()}")
    
    # Selección de features
    st.subheader("Características")
    
    all_cols = [c for c in df.columns if c != target_column]
    numeric_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[all_cols].select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_numeric = st.multiselect(
            "Variables numéricas:",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))] if numeric_cols else []
        )
    
    with col2:
        selected_categorical = st.multiselect(
            "Variables categóricas:",
            categorical_cols,
            default=categorical_cols[:min(1, len(categorical_cols))] if categorical_cols else []
        )
    
    selected_features = selected_numeric + selected_categorical
    
    if not selected_features:
        st.warning("Selecciona al menos una característica")
        st.stop()
    
    # Configuración del modelo
    st.subheader("Configuración")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Tamaño de prueba:", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        model_type = st.selectbox(
            "Algoritmo:",
            ["Random Forest", "Gradient Boosting", "Logistic Regression", "Decision Tree"]
        )
    
    with col3:
        handle_imbalance = st.selectbox(
            "Manejo de desbalance:",
            ["Balanceo automático", "Sin balanceo"]
        )
    
    # Botón de entrenamiento
    if st.button("Entrenar Modelo", type="primary", use_container_width=True):
        with st.spinner("Entrenando..."):
            try:
                X = df[selected_features].copy()
                y = df[target_column].copy()
                
                # Asegurar que y sea numérico
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                # Codificar categóricas
                encoders = {}
                for col in selected_categorical:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le
                
                # Escalar
                scaler = StandardScaler()
                if selected_numeric:
                    X[selected_numeric] = scaler.fit_transform(X[selected_numeric])
                
                # Dividir
                stratify = y if y.nunique() <= 10 else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=stratify
                )
                
                # Modelo
                if model_type == "Random Forest":
                    if handle_imbalance == "Balanceo automático":
                        model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
                    else:
                        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
                elif model_type == "Logistic Regression":
                    if handle_imbalance == "Balanceo automático":
                        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
                    else:
                        model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    if handle_imbalance == "Balanceo automático":
                        model = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
                    else:
                        model = DecisionTreeClassifier(max_depth=10, random_state=42)
                
                model.fit(X_train, y_train)
                
                # Predicciones
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                
                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.0
                
                # Guardar
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
                    'y_pred_proba': y_pred_proba,
                    'class_names': ['No Fraude', 'Fraude']
                }
                
                st.session_state.model_trained = True
                st.session_state.step = 3
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== PASO 3: RESULTADOS ====================
if st.session_state.model_trained and st.session_state.step == 3:
    st.header("Paso 3: Resultados del Modelo")
    
    models = st.session_state.models
    
    # Información del modelo
    st.info(f"Modelo: {models['model_type']} | Target: {models['target_column']}")
    
    # Métricas en tarjetas con texto visible
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        color = "#2ecc71" if models['accuracy'] > 0.8 else "#f39c12" if models['accuracy'] > 0.6 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h3>ACCURACY</h3>
            <h2 style="color:{color}">{models['accuracy']:.1%}</h2>
            <p>Exactitud general</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#2ecc71" if models['precision'] > 0.8 else "#f39c12" if models['precision'] > 0.6 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h3>PRECISION</h3>
            <h2 style="color:{color}">{models['precision']:.1%}</h2>
            <p>Verdaderos positivos / total positivos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color_recall = "#2ecc71" if models['recall'] > 0.7 else "#f39c12" if models['recall'] > 0.4 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h3>RECALL</h3>
            <h2 style="color:{color_recall}">{models['recall']:.1%}</h2>
            <p>Fraudes detectados / total fraudes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        color = "#2ecc71" if models['f1'] > 0.8 else "#f39c12" if models['f1'] > 0.6 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h3>F1 SCORE</h3>
            <h2 style="color:{color}">{models['f1']:.1%}</h2>
            <p>Media armónica precision/recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        color = "#2ecc71" if models['auc'] > 0.8 else "#f39c12" if models['auc'] > 0.6 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h3>ROC AUC</h3>
            <h2 style="color:{color}">{models['auc']:.1%}</h2>
            <p>Capacidad de discriminación</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Diagnóstico
    st.markdown("---")
    if models['recall'] < 0.5:
        st.warning("⚠️ El Recall es bajo. El modelo no está detectando suficientes fraudes.")
        st.markdown("""
        **Cómo mejorar la detección de fraudes:**
        1. Selecciona 'Balanceo automático' en configuración
        2. Usa Random Forest o Gradient Boosting
        3. Agrega más features relacionadas con fraude
        4. Aumenta la tasa de fraude en los datos de ejemplo
        """)
    elif models['accuracy'] > 0.9:
        st.success("✅ Excelente rendimiento. El modelo detecta bien los fraudes.")
    elif models['accuracy'] > 0.8:
        st.success("✅ Buen rendimiento. El modelo funciona correctamente.")
    else:
        st.info("📊 El modelo tiene margen de mejora. Prueba diferentes features o algoritmos.")
    
    # Matriz de Confusión con etiquetas visibles
    st.subheader("Matriz de Confusión")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(models['y_test'], models['y_pred'])
        
        # Asegurar que la matriz sea 2x2
        if cm.shape == (2, 2):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Fraude', 'Fraude'],
                       yticklabels=['No Fraude', 'Fraude'],
                       ax=ax, cbar=False, annot_kws={'size': 12})
            
            # Agregar texto explicativo dentro de la matriz
            ax.text(0.5, -0.15, 'Predicción', transform=ax.transAxes, ha='center', fontsize=10)
            ax.text(-0.25, 0.5, 'Real', transform=ax.transAxes, va='center', rotation=90, fontsize=10)
            
            st.pyplot(fig)
            plt.close()
            
            # Mostrar interpretación
            st.markdown("""
            **Interpretación:**
            - **Esquina superior izquierda:** Verdaderos Negativos (correctos no-fraude)
            - **Esquina superior derecha:** Falsos Positivos (alertas falsas)
            - **Esquina inferior izquierda:** Falsos Negativos (fraudes no detectados)
            - **Esquina inferior derecha:** Verdaderos Positivos (fraudes detectados)
            """)
    
    with col2:
        if models['auc'] > 0.5:
            fig, ax = plt.subplots(figsize=(5, 4))
            fpr, tpr, _ = roc_curve(models['y_test'], models['y_pred_proba'])
            ax.plot(fpr, tpr, label=f'AUC = {models["auc"]:.3f}', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlabel('Tasa de Falsos Positivos', fontsize=10)
            ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=10)
            ax.set_title('Curva ROC', fontsize=12)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    # Botones
    st.markdown("---")
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
    
    st.info("Ingresa los valores para predecir si la transacción es fraudulenta")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        input_data = {}
        
        with col1:
            st.markdown("**Variables Numéricas**")
            for feature in models['selected_numeric']:
                default_val = float(df[feature].mean()) if feature in df.columns else 0.0
                input_data[feature] = st.number_input(
                    f"{feature}:",
                    value=default_val,
                    step=1000.0 if 'amount' in feature else 1.0,
                    format="%.2f",
                    key=f"num_{feature}"
                )
        
        with col2:
            if models['selected_categorical']:
                st.markdown("**Variables Categóricas**")
                for feature in models['selected_categorical']:
                    if feature in df.columns:
                        unique_values = df[feature].dropna().unique().tolist()
                        input_data[feature] = st.selectbox(
                            f"{feature}:",
                            unique_values,
                            key=f"cat_{feature}"
                        )
        
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
                
                st.markdown("---")
                
                if hasattr(models['model'], "predict_proba"):
                    proba = models['model'].predict_proba(input_df)[0]
                    prob_fraude = proba[1] if len(proba) > 1 else proba[0]
                    
                    if prediction == 1:
                        st.error(f"""
                        ### PREDICCION: FRAUDE
                        
                        | Métrica | Valor |
                        |---|---|
                        | **Predicción** | Fraude |
                        | **Confianza** | {prob_fraude:.1%} |
                        | **Modelo** | {models['model_type']} |
                        | **Recall del modelo** | {models['recall']:.1%} |
                        """)
                    else:
                        st.success(f"""
                        ### PREDICCION: LEGITIMO
                        
                        | Métrica | Valor |
                        |---|---|
                        | **Predicción** | Legitimo |
                        | **Confianza** | {1-prob_fraude:.1%} |
                        | **Modelo** | {models['model_type']} |
                        | **Precisión del modelo** | {models['precision']:.1%} |
                        """)
                else:
                    if prediction == 1:
                        st.error(f"### Predicción: FRAUDE")
                    else:
                        st.success(f"### Predicción: LEGITIMO")
                
            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
    
    st.markdown("---")
    if st.button("Volver a Resultados"):
        st.session_state.step = 3
        st.rerun()

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;font-size:12px'>Sistema de Detección de Fraude Bancario</p>", unsafe_allow_html=True)

# app.py - Sistema de Detección de Fraude Bancario
# VERSIÓN CORREGIDA: Naive Bayes + Métricas legibles + Gráficas pequeñas

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, 
                           roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo para gráficas más pequeñas
st.markdown("""
<style>
    [data-testid="stImage"] {
        max-width: 100%;
    }
    .stPlotlyChart {
        max-width: 100%;
    }
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
    }
    .metric-card h2 {
        font-size: 1.5rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INICIALIZACIÓN DE ESTADO ====================
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.df = None
    st.session_state.models = None
    st.session_state.step = 1

# ==================== TÍTULO ====================
st.title("🔍 Sistema de Detección de Fraude Bancario")
st.markdown("---")

# ==================== PASO 1: CARGA DE DATOS ====================
st.header("📊 Paso 1: Carga de Datos")

with st.container():
    data_source = st.radio(
        "Selecciona la fuente de datos:",
        ["📊 Datos de Ejemplo", "📁 Subir CSV", "🌐 Kaggle Dataset"],
        horizontal=True,
        key="data_source"
    )

if st.session_state.df is None:
    
    if data_source == "📊 Datos de Ejemplo":
        if st.button("🎲 Generar datos de ejemplo", use_container_width=True):
            with st.spinner("Generando datos..."):
                try:
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
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    elif data_source == "📁 Subir CSV":
        uploaded_file = st.file_uploader("Arrastra o selecciona un archivo CSV", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if df is not None and len(df) > 0:
                    st.session_state.df = df
                    st.session_state.step = 2
                    st.success(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
                    st.rerun()
                else:
                    st.error("❌ El archivo está vacío")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    else:
        st.info("🔗 Para usar datasets de Kaggle:")
        st.markdown("""
        1. Ve a [Kaggle Datasets](https://www.kaggle.com/datasets)
        2. Copia el path (ejemplo: `mlg-od/credit-card-fraud`)
        3. Pega el path abajo
        """)
        kaggle_path = st.text_input("Path del dataset:", placeholder="mlg-od/credit-card-fraud")
        
        if st.button("📥 Cargar desde Kaggle", use_container_width=True):
            if kaggle_path:
                with st.spinner("Descargando..."):
                    try:
                        import kagglehub
                        path = kagglehub.dataset_download(kaggle_path)
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                if file.endswith('.csv'):
                                    df = pd.read_csv(os.path.join(root, file))
                                    st.session_state.df = df
                                    st.session_state.step = 2
                                    st.success(f"✅ Dataset cargado: {df.shape[0]} filas")
                                    st.rerun()
                                    break
                            if st.session_state.df is not None:
                                break
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Ingresa un path válido")

# ==================== PASO 2: SELECCIÓN DE FEATURES ====================
if st.session_state.df is not None and st.session_state.step == 2 and not st.session_state.model_trained:
    st.header("📊 Paso 2: Configuración del Modelo")
    
    df = st.session_state.df
    
    # Vista previa
    with st.expander("🔍 Vista previa de los datos"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Selección de columna objetivo
    st.subheader("🎯 Selecciona la columna objetivo")
    
    target_column = st.selectbox(
        "Columna objetivo:",
        df.columns
    )
    
    # Mostrar información
    unique_values = df[target_column].nunique()
    st.write(f"**Valores únicos:** {unique_values}")
    
    if unique_values <= 5:
        value_counts = df[target_column].value_counts()
        for val, count in value_counts.items():
            pct = (count / len(df)) * 100
            st.write(f"- {val}: {count} ({pct:.1f}%)")
    
    # Opción para convertir a binario
    if unique_values > 2:
        convert_to_binary = st.checkbox("Convertir a binario (0/1)")
        if convert_to_binary:
            most_frequent = df[target_column].mode()[0]
            st.info(f"Valor más frecuente '{most_frequent}' → 0, otros → 1")
    
    # Selección de features
    st.subheader("🔧 Características")
    
    all_cols = df.columns.tolist()
    if target_column in all_cols:
        all_cols.remove(target_column)
    
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
        if categorical_cols:
            selected_categorical = st.multiselect(
                "Variables categóricas:",
                categorical_cols,
                default=categorical_cols[:min(1, len(categorical_cols))] if categorical_cols else []
            )
        else:
            selected_categorical = []
    
    selected_features = selected_numeric + selected_categorical
    
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una característica")
        st.stop()
    
    # Configuración del modelo
    st.subheader("⚙️ Configuración")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        test_size = st.slider("Tamaño de prueba:", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        model_type = st.selectbox(
            "Algoritmo:",
            ["Random Forest", "Logistic Regression", "Decision Tree", "Naive Bayes"]
        )
    
    with col3:
        reduction_method = st.selectbox(
            "Reducción:",
            ["Ninguna", "PCA", "LDA"]
        )
    
    with col4:
        if reduction_method != "Ninguna":
            n_components = st.number_input("Componentes:", 2, 10, 3)
        else:
            n_components = 3
    
    # Botón de entrenamiento
    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
        with st.spinner("Entrenando..."):
            try:
                X = df[selected_features].copy()
                y = df[target_column].copy()
                
                # Convertir target si es necesario
                if 'convert_to_binary' in locals() and convert_to_binary and unique_values > 2:
                    most_frequent = df[target_column].mode()[0]
                    y = (y != most_frequent).astype(int)
                
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
                
                # Reducción
                reduction_obj = None
                if reduction_method == "PCA":
                    n_comp = min(n_components, len(selected_features), X_train.shape[1])
                    reduction_obj = PCA(n_components=n_comp, random_state=42)
                    X_train = reduction_obj.fit_transform(X_train)
                    X_test = reduction_obj.transform(X_test)
                
                elif reduction_method == "LDA":
                    try:
                        max_components = min(len(np.unique(y_train)) - 1, X_train.shape[1])
                        if max_components >= 1:
                            n_comp = min(n_components, max_components)
                            reduction_obj = LDA(n_components=n_comp)
                            X_train = reduction_obj.fit_transform(X_train, y_train)
                            X_test = reduction_obj.transform(X_test)
                        else:
                            st.warning("⚠️ LDA no disponible, usando datos originales")
                            reduction_method = "Ninguna"
                    except Exception as e:
                        st.warning(f"⚠️ LDA no disponible: {str(e)[:50]}")
                        reduction_method = "Ninguna"
                
                # Modelo
                if model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                elif model_type == "Logistic Regression":
                    model = LogisticRegression(random_state=42, max_iter=1000)
                elif model_type == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=10, random_state=42)
                else:  # Naive Bayes
                    if reduction_method != "Ninguna" or len(selected_numeric) > 0:
                        model = GaussianNB()
                    else:
                        # Para datos categóricos puros
                        model = BernoulliNB()
                
                model.fit(X_train, y_train)
                
                # Predicciones
                y_pred = model.predict(X_test)
                
                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                
                if len(np.unique(y_train)) == 2:
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    try:
                        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                        auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        auc = 0.0
                        y_pred_proba = None
                else:
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    auc = 0.0
                    y_pred_proba = None
                
                # Guardar
                st.session_state.models = {
                    'model': model,
                    'scaler': scaler,
                    'encoders': encoders,
                    'reduction_obj': reduction_obj,
                    'reduction_method': reduction_method,
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
                    'is_binary': len(np.unique(y_train)) == 2,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                st.session_state.model_trained = True
                st.session_state.step = 3
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ==================== PASO 3: MÉTRICAS ====================
if st.session_state.model_trained and st.session_state.step == 3:
    st.header("📊 Paso 3: Resultados del Modelo")
    
    models = st.session_state.models
    
    # Información
    st.info(f"**Modelo:** {models['model_type']} | **Reducción:** {models['reduction_method']} | **Target:** {models['target_column']}")
    
    # Métricas en grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "#2ecc71" if models['accuracy'] > 0.8 else "#f39c12" if models['accuracy'] > 0.6 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2 style="color:{color}">{models['accuracy']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#2ecc71" if models['precision'] > 0.8 else "#f39c12" if models['precision'] > 0.6 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ Precision</h3>
            <h2 style="color:{color}">{models['precision']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = "#2ecc71" if models['recall'] > 0.8 else "#f39c12" if models['recall'] > 0.6 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔍 Recall</h3>
            <h2 style="color:{color}">{models['recall']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        color = "#2ecc71" if models['f1'] > 0.8 else "#f39c12" if models['f1'] > 0.6 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 F1-Score</h3>
            <h2 style="color:{color}">{models['f1']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # ROC-AUC si está disponible
    if models['is_binary'] and models['auc'] > 0:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            color = "#2ecc71" if models['auc'] > 0.8 else "#f39c12" if models['auc'] > 0.6 else "#e74c3c"
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 ROC-AUC</h3>
                <h2 style="color:{color}">{models['auc']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Diagnóstico
    st.markdown("---")
    if models['accuracy'] > 0.9:
        st.success("✅ **Excelente** - El modelo tiene un rendimiento excepcional")
    elif models['accuracy'] > 0.8:
        st.success("✅ **Bueno** - El modelo funciona correctamente")
    elif models['accuracy'] > 0.7:
        st.warning("⚠️ **Aceptable** - El modelo tiene margen de mejora")
    else:
        st.error("❌ **Mejorable** - Considera cambiar features o algoritmo")
    
    # Gráficas más pequeñas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Matriz de Confusión**")
        fig, ax = plt.subplots(figsize=(4, 3))
        cm = confusion_matrix(models['y_test'], models['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicción', fontsize=8)
        ax.set_ylabel('Real', fontsize=8)
        ax.tick_params(labelsize=8)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        if models['is_binary'] and models['y_pred_proba'] is not None:
            st.markdown("**Curva ROC**")
            fig, ax = plt.subplots(figsize=(4, 3))
            fpr, tpr, _ = roc_curve(models['y_test'], models['y_pred_proba'])
            ax.plot(fpr, tpr, label=f'AUC = {models["auc"]:.3f}')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate', fontsize=8)
            ax.set_ylabel('True Positive Rate', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close()
    
    # Botones
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reiniciar", use_container_width=True):
            for key in ['df', 'model_trained', 'models', 'step']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("🔍 Hacer Predicciones", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()

# ==================== PASO 4: PREDICCIONES ====================
if st.session_state.model_trained and st.session_state.step == 4:
    st.header("🔍 Paso 4: Predicción")
    
    models = st.session_state.models
    df = st.session_state.df
    
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
                    step=0.1,
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
        
        submitted = st.form_submit_button("🔍 Predecir", use_container_width=True, type="primary")
        
        if submitted:
            try:
                input_df = pd.DataFrame([input_data])
                
                for col in models['selected_categorical']:
                    if col in models['encoders']:
                        input_df[col] = models['encoders'][col].transform(input_df[col].astype(str))
                
                if models['selected_numeric']:
                    input_df[models['selected_numeric']] = models['scaler'].transform(input_df[models['selected_numeric']])
                
                input_df = input_df[models['selected_features']]
                
                if models['reduction_method'] != "Ninguna" and models['reduction_obj'] is not None:
                    input_df = models['reduction_obj'].transform(input_df)
                
                prediction = models['model'].predict(input_df)[0]
                
                st.markdown("---")
                
                if models['is_binary'] and hasattr(models['model'], "predict_proba"):
                    proba = models['model'].predict_proba(input_df)[0]
                    prob_positive = proba[1] if len(proba) > 1 else proba[0]
                    
                    if prediction == 1:
                        st.error(f"""
                        ### 🚨 Predicción: FRAUDE
                        
                        | Métrica | Valor |
                        |---|---|
                        | **Predicción** | Fraude (Clase 1) |
                        | **Confianza** | {prob_positive:.1%} |
                        | **Modelo** | {models['model_type']} |
                        """)
                    else:
                        st.success(f"""
                        ### ✅ Predicción: LEGÍTIMO
                        
                        | Métrica | Valor |
                        |---|---|
                        | **Predicción** | Legítimo (Clase 0) |
                        | **Confianza** | {1-prob_positive:.1%} |
                        | **Modelo** | {models['model_type']} |
                        """)
                else:
                    st.info(f"""
                    ### 📊 Resultado
                    
                    **Predicción:** {prediction}
                    **Modelo:** {models['model_type']}
                    """)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    if st.button("← Volver"):
        st.session_state.step = 3
        st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;font-size:12px'>Sistema de Detección de Fraude | Streamlit</p>", unsafe_allow_html=True)

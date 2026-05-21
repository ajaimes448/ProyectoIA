# app.py - Sistema de Detección de Fraude Bancario
# VERSIÓN CORREGIDA: LDA agregado + Target flexible

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
st.header("📊 PASO 1: Carga de Datos")

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
                    st.error(f"❌ Error al generar datos: {str(e)}")
    
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
                    st.error("❌ El archivo está vacío o es inválido")
            except Exception as e:
                st.error(f"❌ Error al leer el archivo: {str(e)}")
    
    else:  # Kaggle
        st.info("🔗 Para usar datasets de Kaggle:")
        st.markdown("""
        1. Ve a [Kaggle Datasets](https://www.kaggle.com/datasets)
        2. Copia el path del dataset (ejemplo: `mlg-od/credit-card-fraud`)
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
                        st.error(f"❌ Error al cargar: {str(e)}")
            else:
                st.warning("⚠️ Ingresa un path válido")

# ==================== PASO 2: SELECCIÓN DE FEATURES ====================
if st.session_state.df is not None and st.session_state.step == 2 and not st.session_state.model_trained:
    st.header("📊 PASO 2: Configuración del Modelo")
    
    df = st.session_state.df
    
    # Vista previa
    with st.expander("🔍 Vista previa de los datos"):
        st.dataframe(df.head(10))
        st.write("**Estadísticas básicas:**")
        st.dataframe(df.describe())
    
    # ==================== SELECCIÓN DE COLUMNA OBJETIVO ====================
    st.subheader("🎯 Selecciona la columna objetivo (lo que quieres predecir)")
    
    st.info("💡 Puedes seleccionar CUALQUIER columna como objetivo, no solo 'fraud_flag'")
    
    target_column = st.selectbox(
        "Columna objetivo:",
        df.columns,
        help="Selecciona la columna que quieres predecir (puede ser fraude, categoría, etc.)"
    )
    
    # Mostrar información de la columna seleccionada
    unique_values = df[target_column].nunique()
    st.write(f"**Información:** {unique_values} valores únicos")
    
    # Mostrar distribución
    if unique_values <= 10:  # Solo mostrar si son pocos valores
        st.write("**Distribución:**")
        value_counts = df[target_column].value_counts()
        for val, count in value_counts.items():
            pct = (count / len(df)) * 100
            st.write(f"- {val}: {count} ({pct:.1f}%)")
    
    # Opción para convertir a binario si tiene más de 2 valores
    if unique_values > 2:
        st.warning(f"⚠️ La columna '{target_column}' tiene {unique_values} valores diferentes")
        convert_to_binary = st.checkbox("¿Convertir a binario? (0 = valor más frecuente, 1 = otros)")
        
        if convert_to_binary:
            most_frequent = df[target_column].mode()[0]
            st.info(f"Convertiré: {most_frequent} → 0, Otros valores → 1")
    
    # ==================== SELECCIÓN DE FEATURES ====================
    st.subheader("🔧 Características (Features)")
    
    # Separar tipos de columnas
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
            default=numeric_cols[:min(5, len(numeric_cols))] if numeric_cols else []
        )
    
    with col2:
        if categorical_cols:
            selected_categorical = st.multiselect(
                "Variables categóricas:",
                categorical_cols,
                default=categorical_cols[:min(2, len(categorical_cols))] if categorical_cols else []
            )
        else:
            selected_categorical = []
            st.info("No hay variables categóricas en el dataset")
    
    selected_features = selected_numeric + selected_categorical
    
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una característica")
        st.stop()
    
    st.success(f"✅ Features seleccionadas: {len(selected_features)}")
    
    # ==================== CONFIGURACIÓN DEL MODELO ====================
    st.subheader("⚙️ Configuración del Algoritmo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        test_size = st.slider("Tamaño de prueba:", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        model_type = st.selectbox(
            "Algoritmo:",
            ["Random Forest", "Logistic Regression", "Decision Tree"]
        )
    
    with col3:
        reduction_method = st.selectbox(
            "Reducción de dimensionalidad:",
            ["Ninguna", "PCA", "LDA"]
        )
    
    with col4:
        if reduction_method != "Ninguna":
            n_components = st.slider("Componentes:", 2, 10, 3)
        else:
            n_components = 3
    
    # Botón de entrenamiento
    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo..."):
            try:
                # Preparar datos
                X = df[selected_features].copy()
                y = df[target_column].copy()
                
                # Convertir target a binario si es necesario
                if 'convert_to_binary' in locals() and convert_to_binary and unique_values > 2:
                    most_frequent = df[target_column].mode()[0]
                    y = (y != most_frequent).astype(int)
                    st.info(f"✅ Target convertido a binario: 0={most_frequent}, 1=otros")
                
                # Codificar categóricas
                encoders = {}
                for col in selected_categorical:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le
                
                # Escalar numéricas
                scaler = StandardScaler()
                if selected_numeric:
                    X[selected_numeric] = scaler.fit_transform(X[selected_numeric])
                
                # Dividir
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() <= 10 else None
                )
                
                # Aplicar reducción de dimensionalidad
                reduction_obj = None
                
                if reduction_method == "PCA":
                    reduction_obj = PCA(n_components=min(n_components, len(selected_features), X_train.shape[1]))
                    X_train = reduction_obj.fit_transform(X_train)
                    X_test = reduction_obj.transform(X_test)
                    st.info(f"✅ PCA aplicado: {X_train.shape[1]} componentes")
                
                elif reduction_method == "LDA":
                    try:
                        n_lda = min(n_components, len(np.unique(y_train)) - 1, X_train.shape[1])
                        if n_lda >= 1:
                            reduction_obj = LDA(n_components=n_lda)
                            X_train = reduction_obj.fit_transform(X_train, y_train)
                            X_test = reduction_obj.transform(X_test)
                            st.info(f"✅ LDA aplicado: {X_train.shape[1]} componentes")
                        else:
                            st.warning("⚠️ No se puede aplicar LDA (problema con las dimensiones)")
                            reduction_method = "Ninguna"
                    except Exception as e:
                        st.warning(f"⚠️ LDA no disponible: {str(e)}")
                        reduction_method = "Ninguna"
                
                # Modelo
                if model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                elif model_type == "Logistic Regression":
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    model = DecisionTreeClassifier(max_depth=10, random_state=42)
                
                model.fit(X_train, y_train)
                
                # Predicciones
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") and len(np.unique(y_train)) == 2 else y_pred
                
                # Métricas
                if len(np.unique(y_train)) == 2:
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        auc = 0.0
                else:
                    # Métricas para multiclase
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    auc = 0.0
                
                # Guardar en sesión
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
                    'y_pred_proba': y_pred_proba if len(np.unique(y_train)) == 2 else None
                }
                
                st.session_state.model_trained = True
                st.session_state.step = 3
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento")
                st.info(f"**Detalle:** {str(e)}")
                st.info("**Solución sugerida:** Revisa que las columnas seleccionadas no tengan valores nulos")

# ==================== PASO 3: MÉTRICAS Y EVALUACIÓN ====================
if st.session_state.model_trained and st.session_state.step == 3:
    st.header("📊 PASO 3: Evaluación del Modelo")
    
    models = st.session_state.models
    
    # Mostrar información del modelo
    st.info(f"""
    **Configuración del modelo:**
    - Algoritmo: {models['model_type']}
    - Reducción: {models['reduction_method']}
    - Target: {models['target_column']}
    - Features: {len(models['selected_features'])}
    """)
    
    # Mostrar métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "green" if models['accuracy'] > 0.8 else "orange" if models['accuracy'] > 0.6 else "red"
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:1rem; border-radius:10px; text-align:center">
            <h3>🎯 Accuracy</h3>
            <h2 style="color:{color}">{models['accuracy']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "green" if models['precision'] > 0.8 else "orange" if models['precision'] > 0.6 else "red"
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:1rem; border-radius:10px; text-align:center">
            <h3>⚡ Precision</h3>
            <h2 style="color:{color}">{models['precision']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = "green" if models['recall'] > 0.8 else "orange" if models['recall'] > 0.6 else "red"
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:1rem; border-radius:10px; text-align:center">
            <h3>🔍 Recall</h3>
            <h2 style="color:{color}">{models['recall']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        color = "green" if models['f1'] > 0.8 else "orange" if models['f1'] > 0.6 else "red"
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:1rem; border-radius:10px; text-align:center">
            <h3>📊 F1-Score</h3>
            <h2 style="color:{color}">{models['f1']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    if models['is_binary'] and models['auc'] > 0:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            color = "green" if models['auc'] > 0.8 else "orange" if models['auc'] > 0.6 else "red"
            st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:1rem; border-radius:10px; text-align:center">
                <h3>📈 ROC-AUC</h3>
                <h2 style="color:{color}">{models['auc']:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Diagnóstico
    st.markdown("---")
    st.subheader("💡 Diagnóstico del Modelo")
    
    if models['accuracy'] > 0.9:
        st.success("✅ **EXCELENTE** - El modelo tiene un rendimiento excepcional")
    elif models['accuracy'] > 0.8:
        st.success("✅ **BUENO** - El modelo funciona correctamente")
    elif models['accuracy'] > 0.7:
        st.warning("⚠️ **ACEPTABLE** - El modelo tiene margen de mejora")
    elif models['accuracy'] > 0.6:
        st.warning("⚠️ **REGULAR** - Considera mejorar los features o probar otro modelo")
    else:
        st.error("❌ **MALO** - El modelo no está funcionando bien. Revisa los datos")
    
    # Matriz de confusión
    st.subheader("📊 Matriz de Confusión")
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(models['y_test'], models['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    st.pyplot(fig)
    plt.close()
    
    # Curva ROC si es binario
    if models['is_binary'] and models['y_pred_proba'] is not None:
        st.subheader("📈 Curva ROC")
        fpr, tpr, _ = roc_curve(models['y_test'], models['y_pred_proba'])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f'ROC (AUC = {models["auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    # Botones de navegación
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reiniciar con nuevos datos", use_container_width=True):
            for key in ['df', 'model_trained', 'models', 'step']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("🔍 Ir a Predicciones", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()

# ==================== PASO 4: PREDICCIONES ====================
if st.session_state.model_trained and st.session_state.step == 4:
    st.header("🔍 PASO 4: Predicción de Nuevos Casos")
    
    models = st.session_state.models
    df = st.session_state.df
    
    st.info(f"📌 Prediciendo: {models['target_column']}")
    
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        with col1:
            st.markdown("**📊 Variables Numéricas**")
            for feature in models['selected_numeric']:
                default_val = float(df[feature].mean()) if feature in df.columns else 0.0
                input_data[feature] = st.number_input(
                    f"📌 {feature}:",
                    value=default_val,
                    step=0.01 if df[feature].dtype == float else 1.0,
                    format="%.2f" if df[feature].dtype == float else None,
                    key=f"num_{feature}"
                )
        
        with col2:
            if models['selected_categorical']:
                st.markdown("**🏷️ Variables Categóricas**")
                for feature in models['selected_categorical']:
                    if feature in df.columns:
                        unique_values = df[feature].dropna().unique().tolist()
                        input_data[feature] = st.selectbox(
                            f"📌 {feature}:",
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
                
                if models['is_binary']:
                    probability = models['model'].predict_proba(input_df)[0][1]
                    st.markdown("---")
                    
                    if prediction == 1:
                        st.error(f"""
                        ### 🚨 PREDICCIÓN: CLASE 1
                        
                        | Métrica | Valor |
                        |---------|-------|
                        | **Predicción** | Clase 1 |
                        | **Confianza** | {probability:.2%} |
                        | **Modelo** | {models['model_type']} |
                        | **Precisión** | {models['accuracy']:.2%} |
                        """)
                    else:
                        st.success(f"""
                        ### ✅ PREDICCIÓN: CLASE 0
                        
                        | Métrica | Valor |
                        |---------|-------|
                        | **Predicción** | Clase 0 |
                        | **Confianza** | {1-probability:.2%} |
                        | **Modelo** | {models['model_type']} |
                        | **Precisión** | {models['accuracy']:.2%} |
                        """)
                else:
                    st.markdown("---")
                    st.info(f"""
                    ### 📊 Resultado de la Predicción
                    
                    **Predicción:** {prediction}
                    **Modelo:** {models['model_type']}
                    **Precisión del modelo:** {models['accuracy']:.2%}
                    """)
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")
    
    st.markdown("---")
    if st.button("← Volver a métricas", use_container_width=True):
        st.session_state.step = 3
        st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px">
    <p>Sistema de Detección de Fraude Bancario | Desarrollado con Streamlit</p>
</div>
""", unsafe_allow_html=True)

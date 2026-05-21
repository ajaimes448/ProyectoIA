# app.py - Sistema de Detección de Fraude Bancario
# FLUJO COMPLETO: Carga de datos → Selección de features → Entrenamiento → Métricas → Predicción

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, 
                           roc_curve, mean_squared_error, r2_score)
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

# Estilo personalizado
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header"><h1>🔍 Sistema de Detección de Fraude Bancario</h1><p>Machine Learning para identificar transacciones fraudulentas</p></div>', unsafe_allow_html=True)

# ==================== PASO 1: CARGA DE DATOS ====================
st.header("📊 PASO 1: Carga de Datos")

with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        data_source = st.radio(
            "Selecciona la fuente de datos:",
            ["📁 Subir archivo CSV", "🌐 Descargar de Kaggle", "📊 Usar datos de ejemplo"],
            horizontal=True
        )

df = None
dataset_name = None

# Opción 1: Subir CSV
if data_source == "📁 Subir archivo CSV":
    uploaded_file = st.file_uploader("Arrastra o selecciona un archivo CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
        st.success(f"✅ Archivo cargado: {dataset_name}")
        st.info(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

# Opción 2: Descargar de Kaggle
elif data_source == "🌐 Descargar de Kaggle":
    st.markdown("### Instrucciones")
    st.markdown("""
    1. Ve a [Kaggle Datasets](https://www.kaggle.com/datasets)
    2. Copia el path del dataset (ejemplo: `mlg-od/credit-card-fraud`)
    3. Pégalo en el campo de abajo
    """)
    
    kaggle_path = st.text_input("Path del dataset de Kaggle:", placeholder="mlg-od/credit-card-fraud")
    
    if st.button("📥 Descargar y Cargar"):
        try:
            import kagglehub
            with st.spinner("Descargando dataset desde Kaggle..."):
                path = kagglehub.dataset_download(kaggle_path)
                # Buscar archivos CSV
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.csv'):
                            df = pd.read_csv(os.path.join(root, file))
                            dataset_name = file
                            break
                    if df is not None:
                        break
            st.success(f"✅ Dataset cargado: {dataset_name}")
            st.info(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
        except Exception as e:
            st.error(f"Error al cargar: {str(e)}")

# Opción 3: Datos de ejemplo
else:
    if st.button("🎲 Generar datos de ejemplo"):
        with st.spinner("Generando datos sintéticos..."):
            n_samples = st.slider("Número de transacciones:", 1000, 10000, 5000)
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
            
            # Generar target
            fraud_prob = (
                (df['anomaly_score'] > 0.7) * 0.3 +
                (df['login_attempts'] > 5) * 0.2 +
                (df['suspicious_ip'] == 1) * 0.2 +
                (df['international'] == 1) * 0.15 +
                (df['transaction_amount'] > 50000) * 0.15
            )
            df['fraud_flag'] = (np.random.random(n_samples) < fraud_prob).astype(int)
            
            dataset_name = "datos_ejemplo"
            st.success(f"✅ Datos generados: {n_samples} transacciones")
            st.info(f"Tasa de fraude: {df['fraud_flag'].mean():.2%}")

# Verificar si hay datos
if df is None:
    st.info("👈 Selecciona una fuente de datos para comenzar")
    st.stop()

# ==================== PASO 2: EXPLORACIÓN Y SELECCIÓN ====================
st.header("📊 PASO 2: Exploración y Selección de Features")

# Vista previa
with st.expander("🔍 Vista previa de los datos"):
    st.dataframe(df.head(10))
    st.write("### Estadísticas descriptivas")
    st.dataframe(df.describe())

# Selección de columna objetivo
st.subheader("🎯 Selecciona la columna objetivo (fraude)")

# Detectar posibles columnas de fraude
fraud_candidates = [col for col in df.columns if 'fraud' in col.lower() or 'flag' in col.lower() or 'target' in col.lower()]
if fraud_candidates:
    st.info(f"Columnas detectadas como posibles indicadores de fraude: {', '.join(fraud_candidates)}")

target_column = st.selectbox(
    "Columna que indica si es fraude (1 = Fraude, 0 = No Fraude):",
    df.columns,
    index=df.columns.tolist().index(fraud_candidates[0]) if fraud_candidates else 0
)

# Verificar que la columna objetivo sea binaria
if df[target_column].nunique() > 2:
    st.warning(f"⚠️ La columna '{target_column}' tiene {df[target_column].nunique()} valores únicos. Se recomienda una columna binaria (0/1) para clasificación.")

# Mostrar distribución del target
col1, col2 = st.columns(2)
with col1:
    fraud_count = df[target_column].sum()
    fraud_pct = (fraud_count / len(df)) * 100
    st.metric("Transacciones Fraudulentas", f"{fraud_count:,}")
    st.metric("Porcentaje de Fraude", f"{fraud_pct:.2f}%")

with col2:
    fig, ax = plt.subplots()
    df[target_column].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
    ax.set_title('Distribución de Clases')
    ax.set_xlabel('Clase')
    ax.set_ylabel('Cantidad')
    ax.set_xticklabels(['No Fraude', 'Fraude'] if df[target_column].nunique() == 2 else df[target_column].unique())
    st.pyplot(fig)
    plt.close()

# Selección de features
st.subheader("🔧 Selecciona las características (features) para el modelo")

# Separar features numéricas y categóricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_column in numeric_cols:
    numeric_cols.remove(target_column)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Selección de features numéricas
st.write("### Características Numéricas")
selected_numeric = st.multiselect(
    "Selecciona las variables numéricas:",
    numeric_cols,
    default=numeric_cols[:min(5, len(numeric_cols))]
)

# Selección de features categóricas
if categorical_cols:
    st.write("### Características Categóricas")
    selected_categorical = st.multiselect(
        "Selecciona las variables categóricas:",
        categorical_cols,
        default=categorical_cols[:min(2, len(categorical_cols))]
    )
else:
    selected_categorical = []

# Combinar features seleccionadas
selected_features = selected_numeric + selected_categorical

if not selected_features:
    st.warning("⚠️ Selecciona al menos una característica para entrenar el modelo")
    st.stop()

st.success(f"✅ Features seleccionadas: {len(selected_features)}")

# ==================== PASO 3: ENTRENAMIENTO Y EVALUACIÓN ====================
st.header("🚀 PASO 3: Entrenamiento y Evaluación")

# Configuración del modelo
st.subheader("⚙️ Configuración del Modelo")

col1, col2, col3 = st.columns(3)

with col1:
    test_size = st.slider("Tamaño de prueba:", 0.1, 0.4, 0.2, 0.05)

with col2:
    model_type = st.selectbox(
        "Algoritmo:",
        ["Random Forest", "Logistic Regression", "Decision Tree", "SVM"]
    )

with col3:
    use_pca = st.checkbox("Usar PCA (Reducción de dimensionalidad)")

# Parámetros adicionales según el modelo
if model_type == "Random Forest":
    n_estimators = st.slider("Número de árboles:", 50, 200, 100, 50)
    max_depth = st.slider("Profundidad máxima:", 5, 20, 10, 5)
else:
    n_estimators = 100
    max_depth = 10

# Botón de entrenamiento
if st.button("🎯 Entrenar Modelo", type="primary", use_container_width=True):
    
    with st.spinner("Preprocesando datos y entrenando modelo..."):
        
        # Preparar datos
        X = df[selected_features].copy()
        y = df[target_column].copy()
        
        # Codificar variables categóricas
        encoders = {}
        for col in selected_categorical:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        
        # Escalar características numéricas
        scaler = StandardScaler()
        X[selected_numeric] = scaler.fit_transform(X[selected_numeric])
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Aplicar PCA si está seleccionado
        if use_pca:
            pca = PCA(n_components=min(5, len(selected_features)))
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            st.info(f"PCA aplicado: {len(selected_features)} → {X_train.shape[1]} componentes")
        
        # Seleccionar y entrenar modelo
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:  # SVM
            model = SVC(random_state=42, probability=True)
        
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # ==================== MÉTRICAS DE EVALUACIÓN ====================
        st.markdown("---")
        st.subheader("📈 Resultados y Métricas del Modelo")
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        # Mostrar métricas en tarjetas
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            color_accuracy = "green" if accuracy > 0.8 else "orange" if accuracy > 0.6 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Accuracy</h3>
                <h2 style="color:{color_accuracy}">{accuracy:.2%}</h2>
                <p>Exactitud general</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color_precision = "green" if precision > 0.8 else "orange" if precision > 0.6 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡ Precision</h3>
                <h2 style="color:{color_precision}">{precision:.2%}</h2>
                <p>Precisión del modelo</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            color_recall = "green" if recall > 0.8 else "orange" if recall > 0.6 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔍 Recall</h3>
                <h2 style="color:{color_recall}">{recall:.2%}</h2>
                <p>Sensibilidad (fraudes detectados)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            color_f1 = "green" if f1 > 0.8 else "orange" if f1 > 0.6 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 F1-Score</h3>
                <h2 style="color:{color_f1}">{f1:.2%}</h2>
                <p>Balance precisión/recall</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            color_auc = "green" if auc > 0.8 else "orange" if auc > 0.6 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 ROC-AUC</h3>
                <h2 style="color:{color_auc}">{auc:.2%}</h2>
                <p>Capacidad de discriminación</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Evaluación del modelo
        st.markdown("---")
        st.subheader("📊 Evaluación Detallada del Modelo")
        
        # Determinar si el modelo funcionó bien
        st.markdown("### 🎯 Diagnóstico del Rendimiento")
        
        if accuracy > 0.9:
            st.success("✅ **EXCELENTE** - El modelo tiene un rendimiento excepcional")
        elif accuracy > 0.8:
            st.success("✅ **BUENO** - El modelo funciona correctamente")
        elif accuracy > 0.7:
            st.warning("⚠️ **ACEPTABLE** - El modelo tiene margen de mejora")
        elif accuracy > 0.6:
            st.warning("⚠️ **REGULAR** - Se recomienda mejorar los features o probar otro modelo")
        else:
            st.error("❌ **MALO** - El modelo no está funcionando bien. Revisa los datos o prueba otro algoritmo")
        
        # Recomendaciones basadas en métricas
        st.markdown("### 💡 Recomendaciones")
        recommendations = []
        
        if recall < 0.7:
            recommendations.append("• Aumentar la sensibilidad (recall) para detectar más fraudes")
        if precision < 0.7:
            recommendations.append("• Mejorar la precisión para reducir falsos positivos")
        if accuracy < 0.75:
            recommendations.append("• Considerar más features o un modelo más complejo")
        if auc < 0.7:
            recommendations.append("• El modelo tiene baja capacidad discriminativa - revisar features")
        
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.success("🎉 ¡El modelo tiene un rendimiento excelente en todas las métricas!")
        
        # Matriz de confusión
        st.markdown("### 📊 Matriz de Confusión")
        col1, col2 = st.columns(2)
        
        with col1:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')
            ax.set_title('Matriz de Confusión')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Reporte de clasificación
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2%}"))
        
        # Curva ROC
        if hasattr(model, "predict_proba"):
            st.markdown("### 📈 Curva ROC")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Curva ROC')
            ax.legend()
            st.pyplot(fig)
            plt.close()
        
        # Cross-validation
        st.markdown("### 🔄 Validación Cruzada (5-Fold)")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CV Media", f"{cv_scores.mean():.2%}")
        with col2:
            st.metric("CV Std", f"{cv_scores.std():.2%}")
        with col3:
            st.metric("CV Min/Max", f"{cv_scores.min():.2%} / {cv_scores.max():.2%}")
        
        # Importancia de features (solo para Random Forest)
        if model_type == "Random Forest" and hasattr(model, 'feature_importances_') and not use_pca:
            st.markdown("### 🌟 Importancia de Características")
            
            # Obtener nombres de features
            feature_names = selected_features
            
            # Crear dataframe de importancia
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(importance_df['feature'], importance_df['importance'])
            ax.set_xlabel('Importancia')
            ax.set_title('Importancia de Características')
            st.pyplot(fig)
            plt.close()
        
        # Guardar modelo en session state para predicciones
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['encoders'] = encoders
        st.session_state['selected_features'] = selected_features
        st.session_state['selected_numeric'] = selected_numeric
        st.session_state['selected_categorical'] = selected_categorical
        st.session_state['target_column'] = target_column
        st.session_state['model_trained'] = True
        st.session_state['model_type'] = model_type
        st.session_state['accuracy'] = accuracy
        
        st.balloons()
        st.success("✅ Modelo entrenado y guardado exitosamente!")

# ==================== PASO 4: PREDICCIÓN INDIVIDUAL ====================
if 'model_trained' in st.session_state and st.session_state['model_trained']:
    st.markdown("---")
    st.header("🔍 PASO 4: Predicción de Nuevas Transacciones")
    
    st.info("Ingresa los datos de la transacción para determinar si es fraudulenta")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        # Inputs para features numéricas
        with col1:
            st.markdown("**Variables Numéricas**")
            for feature in st.session_state['selected_numeric']:
                # Obtener valor promedio del dataset como default
                default_val = float(df[feature].mean()) if feature in df.columns else 0
                input_data[feature] = st.number_input(
                    f"{feature}:",
                    value=default_val,
                    step=0.01 if df[feature].dtype == float else 1.0,
                    format="%.2f" if df[feature].dtype == float else None
                )
        
        # Inputs para features categóricas
        with col2:
            st.markdown("**Variables Categóricas**")
            for feature in st.session_state['selected_categorical']:
                if feature in df.columns:
                    unique_values = df[feature].unique().tolist()
                    input_data[feature] = st.selectbox(
                        f"{feature}:",
                        unique_values
                    )
        
        submitted = st.form_submit_button("🔍 Predecir Fraude", use_container_width=True, type="primary")
        
        if submitted:
            # Preparar datos para predicción
            input_df = pd.DataFrame([input_data])
            
            # Codificar categóricas
            for col in st.session_state['selected_categorical']:
                if col in st.session_state['encoders']:
                    input_df[col] = st.session_state['encoders'][col].transform(input_df[col].astype(str))
            
            # Escalar numéricas
            input_df[st.session_state['selected_numeric']] = st.session_state['scaler'].transform(
                input_df[st.session_state['selected_numeric']]
            )
            
            # Asegurar orden de columnas
            input_df = input_df[st.session_state['selected_features']]
            
            # Predicción
            prediction = st.session_state['model'].predict(input_df)[0]
            probability = st.session_state['model'].predict_proba(input_df)[0][1] if hasattr(st.session_state['model'], "predict_proba") else prediction
            
            # Mostrar resultado
            st.markdown("---")
            st.subheader("📋 Resultado de la Predicción")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if prediction == 1:
                    st.error(f"""
                    ### 🚨 ALERTA DE FRAUDE
                    
                    **La transacción ha sido identificada como FRAUDULENTA**
                    
                    - **Confianza del modelo:** {probability:.2%}
                    - **Modelo utilizado:** {st.session_state['model_type']}
                    - **Precisión del modelo:** {st.session_state['accuracy']:.2%}
                    
                    ⚠️ **Recomendación:** Bloquear transacción y solicitar verificación adicional.
                    """)
                else:
                    st.success(f"""
                    ### ✅ TRANSACCIÓN LEGÍTIMA
                    
                    **La transacción parece ser legítima**
                    
                    - **Confianza del modelo:** {1-probability:.2%}
                    - **Modelo utilizado:** {st.session_state['model_type']}
                    - **Precisión del modelo:** {st.session_state['accuracy']:.2%}
                    
                    ✅ **Recomendación:** Procesar transacción normalmente.
                    """)
            
            # Mostrar factores de riesgo si es fraude
            if prediction == 1 and probability > 0.7:
                st.markdown("### ⚠️ Factores de Riesgo Detectados")
                risk_factors = []
                
                if 'transaction_amount' in input_data and input_data['transaction_amount'] > 50000:
                    risk_factors.append("• Monto de transacción inusualmente alto")
                if 'login_attempts' in input_data and input_data['login_attempts'] > 5:
                    risk_factors.append("• Múltiples intentos de inicio de sesión")
                if 'anomaly_score' in input_data and input_data['anomaly_score'] > 0.7:
                    risk_factors.append("• Puntaje de anomalía elevado")
                if 'international' in input_data and input_data['international'] == 1:
                    risk_factors.append("• Transacción internacional")
                if 'suspicious_ip' in input_data and input_data['suspicious_ip'] == 1:
                    risk_factors.append("• IP sospechosa detectada")
                
                for factor in risk_factors:
                    st.write(factor)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Sistema de Detección de Fraude Bancario | Desarrollado con Streamlit y Scikit-learn</p>
</div>
""", unsafe_allow_html=True)

"""
Fraud Detection Dashboard - Modern Streamlit Application
Displays notebook outputs and model predictions for fraud detection models.
v1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, transparent 100%);
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Prediction result cards */
    .prediction-fraud {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .prediction-safe {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Cache data loading functions
@st.cache_data
def load_fraud_data():
    """Load the fraud detection dataset."""
    data_path = project_root / "data" / "processed" / "fraud_featured.parquet"
    if data_path.exists():
        return pd.read_parquet(data_path)
    return None


@st.cache_data
def load_creditcard_data():
    """Load the credit card dataset."""
    data_path = project_root / "data" / "raw" / "creditcard.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return None


@st.cache_resource
def load_models():
    """Load all trained models and their metrics."""
    models = {}
    models_dir = project_root / "models"
    
    model_files = [
        ("fraud_random_forest", "E-commerce Fraud - Random Forest"),
        ("fraud_logistic_regression", "E-commerce Fraud - Logistic Regression"),
        ("creditcard_random_forest", "Credit Card - Random Forest"),
        ("creditcard_logistic_regression", "Credit Card - Logistic Regression"),
    ]
    
    for filename, display_name in model_files:
        model_path = models_dir / f"{filename}.joblib"
        metrics_path = models_dir / f"{filename}_metrics.joblib"
        
        if not model_path.exists():
            continue

        try:
            loaded_model = joblib.load(model_path)
            loaded_metrics = joblib.load(metrics_path) if metrics_path.exists() else {}
        except Exception as e:
            models[display_name] = {
                "model": None,
                "metrics": {},
                "type": "fraud" if "fraud" in filename else "creditcard",
                "load_error": str(e),
            }
            continue

        models[display_name] = {
            "model": loaded_model,
            "metrics": loaded_metrics,
            "type": "fraud" if "fraud" in filename else "creditcard",
        }
    
    return models


def create_gauge_chart(value, title, max_val=1.0):
    """Create a gauge chart for metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        number={'font': {'size': 28}, 'suffix': "", 'valueformat': '.3f'},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, max_val * 0.5], 'color': '#ffebee'},
                {'range': [max_val * 0.5, max_val * 0.75], 'color': '#fff3e0'},
                {'range': [max_val * 0.75, max_val], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "#4caf50", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_confusion_matrix_plot(tn, fp, fn, tp):
    """Create a confusion matrix heatmap."""
    cm = np.array([[tn, fp], [fn, tp]])
    labels = ['Non-Fraud', 'Fraud']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate="%{text:,}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Fraud Detection Dashboard</h1>
        <p>Machine Learning Models for E-commerce and Credit Card Fraud Detection</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/security-checked.png", width=150)
        st.markdown("### Navigation")
        
        page = st.radio(
            "Select Page",
            ["📊 Dashboard Overview", "🔍 Model Performance", "🎯 Make Predictions", "📈 Data Exploration"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard provides:
        - **Model Performance** metrics
        - **Real-time Predictions**
        - **Data Exploration** tools
        - **SHAP Explainability** insights
        """)
        
        st.markdown("---")
        st.markdown("### Models Available")
        models = load_models()
        any_loaded = False
        for name, info in models.items():
            if info.get("model") is not None:
                any_loaded = True
                st.markdown(f"✅ {name}")
            else:
                st.markdown(f"⚠️ {name}")
                err = info.get("load_error")
                if err:
                    with st.expander("Show load error"):
                        st.code(err)

        if not any_loaded:
            st.warning(
                "No models could be loaded in the current Python environment. "
                "This is usually caused by scikit-learn/imbalanced-learn version mismatch with the saved .joblib files. "
                "You can still use Data Exploration, but Predictions and Model Performance will be limited until models load."
            )
        
        return page


def render_dashboard_overview():
    """Render the main dashboard overview."""
    st.markdown('<div class="section-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Load data
    fraud_data = load_fraud_data()
    creditcard_data = load_creditcard_data()
    models = load_models()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = 0
        if fraud_data is not None:
            total_transactions += len(fraud_data)
        if creditcard_data is not None:
            total_transactions += len(creditcard_data)
        st.metric("Total Transactions", f"{total_transactions:,}", delta="Analyzed")
    
    with col2:
        loaded_models_count = sum(1 for info in models.values() if info.get("model") is not None)
        st.metric("Models Loaded", loaded_models_count, delta=f"{len(models)} discovered")
    
    with col3:
        if fraud_data is not None:
            fraud_rate = fraud_data['class'].mean() * 100
            st.metric("E-commerce Fraud Rate", f"{fraud_rate:.2f}%", delta="-0.5%", delta_color="inverse")
    
    with col4:
        if creditcard_data is not None:
            cc_fraud_rate = creditcard_data['Class'].mean() * 100
            st.metric("Credit Card Fraud Rate", f"{cc_fraud_rate:.4f}%", delta="-0.01%", delta_color="inverse")
    
    st.markdown("---")
    
    # Model performance summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏪 E-commerce Fraud Detection")
        if "E-commerce Fraud - Random Forest" in models and models["E-commerce Fraud - Random Forest"].get("model") is not None:
            metrics = models["E-commerce Fraud - Random Forest"]["metrics"]
            
            fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'indicator'}]*3])
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get('auc_pr', 0),
                title={'text': "AUC-PR"},
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#667eea"}}
            ), row=1, col=1)
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get('f1', 0),
                title={'text': "F1-Score"},
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#764ba2"}}
            ), row=1, col=2)
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get('precision', 0),
                title={'text': "Precision"},
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#4caf50"}}
            ), row=1, col=3)
            
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 💳 Credit Card Fraud Detection")
        if "Credit Card - Random Forest" in models and models["Credit Card - Random Forest"].get("model") is not None:
            metrics = models["Credit Card - Random Forest"]["metrics"]
            
            fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'indicator'}]*3])
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get('auc_pr', 0),
                title={'text': "AUC-PR"},
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#667eea"}}
            ), row=1, col=1)
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get('f1', 0),
                title={'text': "F1-Score"},
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#764ba2"}}
            ), row=1, col=2)
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get('precision', 0),
                title={'text': "Precision"},
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#4caf50"}}
            ), row=1, col=3)
            
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    # Dataset overview
    st.markdown("---")
    st.markdown("#### 📁 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if fraud_data is not None:
            st.markdown("**E-commerce Fraud Dataset**")
            
            # Class distribution
            class_counts = fraud_data['class'].value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=['Non-Fraud', 'Fraud'],
                color_discrete_sequence=['#51cf66', '#ff6b6b'],
                hole=0.4
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if creditcard_data is not None:
            st.markdown("**Credit Card Fraud Dataset**")
            
            class_counts = creditcard_data['Class'].value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=['Non-Fraud', 'Fraud'],
                color_discrete_sequence=['#51cf66', '#ff6b6b'],
                hole=0.4
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)


def render_model_performance():
    """Render detailed model performance page."""
    st.markdown('<div class="section-header">🔍 Model Performance Analysis</div>', unsafe_allow_html=True)
    
    models = load_models()

    selectable_models = [name for name, info in models.items() if info.get("model") is not None]
    if not selectable_models:
        st.warning("No models are currently loadable. Please fix environment versions, then refresh the app.")
        return
    
    # Model selector
    selected_model = st.selectbox("Select Model", selectable_models)
    
    model_info = models[selected_model]
    metrics = model_info["metrics"]
    
    st.markdown("---")
    
    # Metrics display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("AUC-PR", f"{metrics.get('auc_pr', 0):.4f}")
    with col2:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
    with col3:
        st.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
    with col4:
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    with col5:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        tn = metrics.get('tn', 0)
        fp = metrics.get('fp', 0)
        fn = metrics.get('fn', 0)
        tp = metrics.get('tp', 0)
        
        fig = create_confusion_matrix_plot(tn, fp, fn, tp)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Metrics radar chart
        categories = ['Precision', 'Recall', 'F1-Score', 'AUC-PR', 'ROC-AUC']
        values = [
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            metrics.get('auc_pr', 0),
            metrics.get('roc_auc', 0)
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='#667eea', width=2),
            name=selected_model
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,
            title="Performance Radar",
            height=350,
            margin=dict(l=60, r=60, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.markdown("---")
    st.markdown("#### 📊 Model Comparison")
    
    comparison_data = []
    for name, info in models.items():
        m = info["metrics"]
        comparison_data.append({
            "Model": name,
            "AUC-PR": m.get('auc_pr', 0),
            "ROC-AUC": m.get('roc_auc', 0),
            "F1-Score": m.get('f1', 0),
            "Precision": m.get('precision', 0),
            "Recall": m.get('recall', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        comparison_df.melt(id_vars=['Model'], var_name='Metric', value_name='Value'),
        x='Model',
        y='Value',
        color='Metric',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def render_predictions():
    """Render the predictions page."""
    st.markdown('<div class="section-header">🎯 Make Predictions</div>', unsafe_allow_html=True)
    
    models = load_models()
    
    # Model type selection
    prediction_type = st.radio(
        "Select Prediction Type",
        ["E-commerce Transaction", "Credit Card Transaction"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if prediction_type == "E-commerce Transaction":
        render_fraud_prediction(models)
    else:
        render_creditcard_prediction(models)


def render_fraud_prediction(models):
    """Render e-commerce fraud prediction form."""
    st.markdown("#### Enter Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        purchase_value = st.number_input("Purchase Value ($)", min_value=0, max_value=10000, value=50)
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
        source = st.selectbox("Traffic Source", ["SEO", "Ads", "Direct"])
    
    with col2:
        hour_of_day = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 3)
        browser = st.selectbox("Browser", ["Chrome", "Safari", "FireFox", "IE", "Opera"])
    
    with col3:
        time_since_signup = st.number_input("Time Since Signup (seconds)", min_value=0, max_value=31536000, value=86400)
        tx_count_1h = st.number_input("Transactions in Last Hour", min_value=0, max_value=100, value=1)
        sex = st.selectbox("Gender", ["M", "F"])
    
    country = st.selectbox("Country", ["United States", "United Kingdom", "Germany", "China", "Unknown"])
    
    if st.button("🔍 Predict Fraud Risk", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame([{
            'purchase_value': purchase_value,
            'age': age,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'time_since_signup': time_since_signup,
            'tx_count_user_id_1h': tx_count_1h,
            'tx_count_user_id_24h': tx_count_1h * 5,
            'user_total_transactions': tx_count_1h * 10,
            'source': source,
            'browser': browser,
            'sex': sex,
            'country': country
        }])
        
        # Get prediction
        model_name = "E-commerce Fraud - Random Forest"
        if model_name in models and models[model_name].get("model") is not None:
            model = models[model_name]["model"]
            
            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                st.markdown("### Prediction Result")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-fraud">
                            <h2>⚠️ FRAUD DETECTED</h2>
                            <p style="font-size: 1.5rem;">Confidence: {proba[1]*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-safe">
                            <h2>✅ TRANSACTION SAFE</h2>
                            <p style="font-size: 1.5rem;">Confidence: {proba[0]*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Probability gauge
                st.markdown("---")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba[1] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Probability (%)"},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff6b6b" if proba[1] > 0.5 else "#51cf66"},
                        'steps': [
                            {'range': [0, 30], 'color': '#e8f5e9'},
                            {'range': [30, 70], 'color': '#fff3e0'},
                            {'range': [70, 100], 'color': '#ffebee'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("E-commerce fraud model is not available (failed to load in this environment).")


def render_creditcard_prediction(models):
    """Render credit card fraud prediction form."""
    st.markdown("#### Enter Transaction Details")
    st.info("💡 Credit card features are PCA-transformed. Enter values for V1-V28, Time, and Amount.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_val = st.number_input("Time (seconds from first tx)", min_value=0, value=0)
        amount = st.number_input("Amount ($)", min_value=0.0, max_value=50000.0, value=100.0)
    
    with col2:
        st.markdown("**Quick Presets:**")
        preset = st.selectbox("Load Example", ["Custom", "Normal Transaction", "Suspicious Transaction"])
    
    # V features
    st.markdown("#### PCA Features (V1-V28)")
    
    if preset == "Normal Transaction":
        v_values = [0.0] * 28
    elif preset == "Suspicious Transaction":
        v_values = [-2.0, 2.0, -1.5, 1.5, -1.0, 1.0, -0.5, 0.5] + [0.0] * 20
    else:
        v_values = [0.0] * 28
    
    cols = st.columns(7)
    v_inputs = []
    for i in range(28):
        with cols[i % 7]:
            v_inputs.append(st.number_input(f"V{i+1}", value=v_values[i], format="%.4f", key=f"v{i+1}"))
    
    if st.button("🔍 Predict Fraud Risk", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame([{
            'Time': time_val,
            **{f'V{i+1}': v_inputs[i] for i in range(28)},
            'Amount': amount
        }])
        
        # Reorder columns
        feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        input_data = input_data[feature_cols]
        
        # Get prediction
        model_name = "Credit Card - Random Forest"
        if model_name in models and models[model_name].get("model") is not None:
            model = models[model_name]["model"]
            
            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                st.markdown("### Prediction Result")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-fraud">
                            <h2>⚠️ FRAUD DETECTED</h2>
                            <p style="font-size: 1.5rem;">Confidence: {proba[1]*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-safe">
                            <h2>✅ TRANSACTION SAFE</h2>
                            <p style="font-size: 1.5rem;">Confidence: {proba[0]*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("Credit card fraud model is not available (failed to load in this environment).")


def render_data_exploration():
    """Render data exploration page."""
    st.markdown('<div class="section-header">📈 Data Exploration</div>', unsafe_allow_html=True)
    
    dataset = st.radio("Select Dataset", ["E-commerce Fraud", "Credit Card Fraud"], horizontal=True)
    
    st.markdown("---")
    
    if dataset == "E-commerce Fraud":
        df = load_fraud_data()
        target_col = 'class'
    else:
        df = load_creditcard_data()
        target_col = 'Class'
    
    if df is None:
        st.warning("Dataset not found.")
        return
    
    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Fraud Cases", f"{df[target_col].sum():,}")
    with col4:
        st.metric("Fraud Rate", f"{df[target_col].mean()*100:.4f}%")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Distributions", "🔗 Correlations"])
    
    with tab1:
        st.dataframe(df.head(100), use_container_width=True)
    
    with tab2:
        if dataset == "E-commerce Fraud":
            col1, col2 = st.columns(2)
            
            with col1:
                # Purchase value distribution
                fig = px.histogram(
                    df, x='purchase_value', color=target_col,
                    title="Purchase Value Distribution",
                    color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
                    barmode='overlay', opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age distribution
                fig = px.histogram(
                    df, x='age', color=target_col,
                    title="Age Distribution",
                    color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
                    barmode='overlay', opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Hour of day
            fig = px.histogram(
                df, x='hour_of_day', color=target_col,
                title="Transactions by Hour of Day",
                color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df, x='Amount', color=target_col,
                    title="Transaction Amount Distribution",
                    color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
                    barmode='overlay', opacity=0.7,
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(
                    df, x='Time', color=target_col,
                    title="Transaction Time Distribution",
                    color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
                    barmode='overlay', opacity=0.7,
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 15:
            # Sample columns for large datasets
            selected_cols = numeric_cols[:15]
            st.info(f"Showing correlation for first 15 numeric columns")
        else:
            selected_cols = numeric_cols
        
        corr_matrix = df[selected_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


# Main app
def main():
    render_header()
    page = render_sidebar()
    
    if page == "📊 Dashboard Overview":
        render_dashboard_overview()
    elif page == "🔍 Model Performance":
        render_model_performance()
    elif page == "🎯 Make Predictions":
        render_predictions()
    elif page == "📈 Data Exploration":
        render_data_exploration()


if __name__ == "__main__":
    main()

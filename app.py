# streamlit_ml_regression_complete.py
"""
COMPLETE ML REGRESSION GUIDE - Streamlit Application
=====================================================
This interactive app demonstrates Simple, Multiple, and Polynomial Regression
with step-by-step explanations, visualizations, and real-time interactions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Regression Complete Guide",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E88E5 0%, #42a5f5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .sub-header {
        font-size: 2rem;
        color: #0D47A1;
        padding: 0.5rem;
        border-bottom: 3px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .step-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #90caf9;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for storing data and models
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_simple_linear_data(n_samples=200, noise=1.5, slope=2, intercept=3):
    """Generate data for simple linear regression"""
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10
    y = slope * X.squeeze() + intercept + np.random.randn(n_samples) * noise
    return X, y, {
        'true_slope': slope,
        'true_intercept': intercept,
        'equation': f'y = {slope}x + {intercept} + noise',
        'feature_names': ['X']
    }

def generate_multiple_linear_data(n_samples=200, noise=2.0):
    """Generate data for multiple linear regression"""
    np.random.seed(42)
    X = np.random.rand(n_samples, 3) * 10
    # y = 3x₁ + 2x₂ - 1.5x₃ + 5 + noise
    y = (3 * X[:, 0] + 2 * X[:, 1] - 1.5 * X[:, 2] + 5 + 
         np.random.randn(n_samples) * noise)
    return X, y, {
        'true_coefficients': [3, 2, -1.5],
        'true_intercept': 5,
        'equation': 'y = 3x₁ + 2x₂ - 1.5x₃ + 5 + noise',
        'feature_names': ['X₁', 'X₂', 'X₃']
    }

def generate_polynomial_data(n_samples=200, noise=3.0, degree=2):
    """Generate data for polynomial regression"""
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10
    if degree == 2:
        y = 0.5 * X.squeeze()**2 - 2 * X.squeeze() + 3 + np.random.randn(n_samples) * noise
        equation = 'y = 0.5x² - 2x + 3 + noise'
    elif degree == 3:
        y = 0.1 * X.squeeze()**3 - 0.5 * X.squeeze()**2 + 2 * X.squeeze() + 1 + np.random.randn(n_samples) * noise
        equation = 'y = 0.1x³ - 0.5x² + 2x + 1 + noise'
    else:
        y = 0.02 * X.squeeze()**4 - 0.3 * X.squeeze()**3 + X.squeeze()**2 - 2 * X.squeeze() + 5 + np.random.randn(n_samples) * noise
        equation = 'y = 0.02x⁴ - 0.3x³ + x² - 2x + 5 + noise'
    
    return X, y, {
        'true_degree': degree,
        'equation': equation,
        'feature_names': ['X']
    }

# ============================================================================
# MAIN APP HEADER
# ============================================================================

st.markdown('<h1 class="main-header">📊 Complete Guide to Regression in Machine Learning</h1>', 
            unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #e3f2fd; border-radius: 10px; margin-bottom: 2rem;'>
        <h3>Learn Simple, Multiple, and Polynomial Regression Step by Step</h3>
        <p>This interactive application demonstrates the complete machine learning pipeline from data preparation to model evaluation.</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - REGRESSION TYPE SELECTION
# ============================================================================

with st.sidebar:
    st.image("https://www.tibco.com/sites/tibco/files/media_entity/2021-05/linear-regression-diagram.svg", 
             width=250, caption="Regression Types")
    
    st.markdown("## 🎯 Select Regression Type")
    regression_type = st.radio(
        "Choose the type of regression you want to explore:",
        ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("## ⚙️ Data Parameters")
    n_samples = st.slider("Number of Samples", 50, 500, 200, 10)
    noise_level = st.slider("Noise Level", 0.0, 5.0, 2.0, 0.1)
    test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
    
    if regression_type == "Simple Linear Regression":
        slope = st.slider("True Slope", -5.0, 5.0, 2.0, 0.1)
        intercept = st.slider("True Intercept", -5.0, 5.0, 3.0, 0.1)
    elif regression_type == "Polynomial Regression":
        poly_degree = st.selectbox("True Polynomial Degree", [2, 3, 4], index=0)
    
    st.markdown("---")
    
    st.markdown("## 🎨 Visualization Options")
    show_residuals = st.checkbox("Show Residual Plots", True)
    show_confidence = st.checkbox("Show Confidence Interval", False)
    
    st.markdown("---")
    
    st.markdown("## 📚 Learning Resources")
    st.info("""
    **Key Concepts:**
    - Linear relationship
    - Multiple features
    - Polynomial features
    - Overfitting
    - R² score
    - RMSE
    """)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Step indicator
steps = ["1️⃣ Data Generation", "2️⃣ Exploratory Analysis", "3️⃣ Data Preparation", 
         "4️⃣ Model Training", "5️⃣ Model Evaluation", "6️⃣ Predictions"]
current_step = st.selectbox("📋 Select Step:", steps, index=0)

st.markdown("---")

# ============================================================================
# STEP 1: DATA GENERATION
# ============================================================================

if current_step == steps[0]:
    st.markdown('<h2 class="sub-header">Step 1: Data Generation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="step-box">
        <h4>📝 Understanding Data Generation</h4>
        <p>We generate synthetic data with known relationships to understand how regression works. 
        This helps us verify if our model can discover the true underlying pattern.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate data based on selection
        if regression_type == "Simple Linear Regression":
            X, y, metadata = generate_simple_linear_data(n_samples, noise_level, slope, intercept)
            st.markdown(f"""
            **📐 True Relationship:**  
            `{metadata['equation']}`
            
            **Parameters:**
            - Slope: {metadata['true_slope']}
            - Intercept: {metadata['true_intercept']}
            """)
            
        elif regression_type == "Multiple Linear Regression":
            X, y, metadata = generate_multiple_linear_data(n_samples, noise_level)
            st.markdown(f"""
            **📐 True Relationship:**  
            `{metadata['equation']}`
            
            **Coefficients:**
            - β₁ (X₁): {metadata['true_coefficients'][0]}
            - β₂ (X₂): {metadata['true_coefficients'][1]}
            - β₃ (X₃): {metadata['true_coefficients'][2]}
            - Intercept: {metadata['true_intercept']}
            """)
            
        else:  # Polynomial Regression
            X, y, metadata = generate_polynomial_data(n_samples, noise_level, poly_degree)
            st.markdown(f"""
            **📐 True Relationship:**  
            `{metadata['equation']}`
            
            **Degree:** {metadata['true_degree']}
            """)
        
        # Store in session state
        st.session_state.datasets[regression_type] = {
            'X': X,
            'y': y,
            'metadata': metadata
        }
        
        st.success("✅ Data generated successfully!")
    
    with col2:
        # Visualize generated data
        st.markdown("### 📊 Generated Data Preview")
        
        if regression_type == "Multiple Linear Regression":
            # For multiple regression, show pairplot
            df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(3)])
            df['y'] = y
            
            fig = px.scatter_matrix(df, dimensions=[f'X{i+1}' for i in range(3)], color=y,
                                   title="Pairwise Relationships",
                                   labels={'color': 'y value'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            # For simple and polynomial, show scatter plot
            fig = px.scatter(x=X.squeeze(), y=y, 
                           title=f"Generated Data Distribution",
                           labels={'x': 'X', 'y': 'y'},
                           trendline="lowess" if regression_type == "Polynomial Regression" else None)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show data statistics
        st.markdown("### 📈 Data Statistics")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Mean of y", f"{y.mean():.2f}")
        with col_stat2:
            st.metric("Std of y", f"{y.std():.2f}")
        with col_stat3:
            st.metric("Range", f"{y.max()-y.min():.2f}")

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

elif current_step == steps[1]:
    st.markdown('<h2 class="sub-header">Step 2: Exploratory Data Analysis (EDA)</h2>', 
                unsafe_allow_html=True)
    
    if regression_type not in st.session_state.datasets:
        st.warning("Please generate data first in Step 1!")
    else:
        data = st.session_state.datasets[regression_type]
        X, y = data['X'], data['y']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="step-box">
            <h4>🔍 What is EDA?</h4>
            <p>Exploratory Data Analysis helps us understand the patterns, relationships, 
            and characteristics of our data before building models.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Basic statistics
            st.markdown("### 📊 Descriptive Statistics")
            df_stats = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [len(y), y.mean(), y.std(), y.min(), 
                         np.percentile(y, 25), np.median(y), 
                         np.percentile(y, 75), y.max()]
            })
            st.dataframe(df_stats, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Distribution Analysis")
            
            # Create distribution plots
            fig = make_subplots(rows=2, cols=2, 
                               subplot_titles=('Target Distribution', 'Box Plot',
                                             'Q-Q Plot', 'Correlation Heatmap'))
            
            # Histogram
            fig.add_trace(go.Histogram(x=y, nbinsx=30, name='Distribution'), row=1, col=1)
            
            # Box plot
            fig.add_trace(go.Box(y=y, name='Box Plot'), row=1, col=2)
            
            # Q-Q plot (simplified)
            from scipy import stats
            theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(y)))
            sample_q = np.sort(y)
            fig.add_trace(go.Scatter(x=theoretical_q, y=sample_q, mode='markers',
                                     name='Q-Q Plot'), row=2, col=1)
            
            # Add diagonal line for Q-Q plot
            min_val = min(theoretical_q.min(), sample_q.min())
            max_val = max(theoretical_q.max(), sample_q.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode='lines', line=dict(dash='dash', color='red'),
                                     showlegend=False), row=2, col=1)
            
            # Correlation heatmap for multiple regression
            if regression_type == "Multiple Linear Regression":
                corr_matrix = np.corrcoef(np.column_stack([X, y]).T)
                fig.add_trace(go.Heatmap(z=corr_matrix, 
                                        x=['X1', 'X2', 'X3', 'y'],
                                        y=['X1', 'X2', 'X3', 'y'],
                                        colorscale='RdBu', zmid=0), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("""
        <div class="insight-box">
        <h4>💡 Key Insights from EDA:</h4>
        <ul>
        <li>The target variable appears to be normally distributed (check Q-Q plot)</li>
        <li>There are no obvious outliers in the data</li>
        <li>The relationship between features and target appears to be {}linear</li>
        </ul>
        </div>
        """.format("non-" if regression_type == "Polynomial Regression" else ""), 
        unsafe_allow_html=True)

# ============================================================================
# STEP 3: DATA PREPARATION
# ============================================================================

elif current_step == steps[2]:
    st.markdown('<h2 class="sub-header">Step 3: Data Preparation</h2>', unsafe_allow_html=True)
    
    if regression_type not in st.session_state.datasets:
        st.warning("Please generate data first in Step 1!")
    else:
        data = st.session_state.datasets[regression_type]
        X, y = data['X'], data['y']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="step-box">
            <h4>🔄 Data Preparation Steps:</h4>
            <ol>
            <li><b>Train-Test Split:</b> Divide data into training (80%) and testing (20%) sets</li>
            <li><b>Feature Scaling:</b> Standardize features to have mean=0 and std=1</li>
            <li><b>Polynomial Features:</b> Create interaction terms (for polynomial regression)</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # Perform data preparation
            test_size_ratio = st.slider("Test Size Ratio", 0.1, 0.4, 0.2, 0.05)
            scale_features = st.checkbox("Apply Feature Scaling", 
                                        value=(regression_type == "Multiple Linear Regression"))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_ratio, random_state=42
            )
            
            # Scale if requested
            scaler = None
            if scale_features and X.shape[1] > 1:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                st.success("✅ Features scaled successfully!")
            
            # Store prepared data
            st.session_state.prepared_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler
            }
            
            # Display data shapes
            st.markdown("### 📊 Data Shapes:")
            st.write(f"Training set: {X_train.shape[0]} samples")
            st.write(f"Testing set: {X_test.shape[0]} samples")
        
        with col2:
            st.markdown("### 📈 Training vs Testing Distribution")
            
            # Plot training and testing distributions
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=y_train, nbinsx=30, name='Training', opacity=0.7))
            fig.add_trace(go.Histogram(x=y_test, nbinsx=30, name='Testing', opacity=0.7))
            fig.update_layout(title="Target Distribution: Training vs Testing",
                            xaxis_title="y value", yaxis_title="Count",
                            barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics comparison
            st.markdown("### 📊 Statistics Comparison:")
            comp_df = pd.DataFrame({
                'Set': ['Training', 'Testing'],
                'Mean': [y_train.mean(), y_test.mean()],
                'Std': [y_train.std(), y_test.std()],
                'Min': [y_train.min(), y_test.min()],
                'Max': [y_train.max(), y_test.max()]
            })
            st.dataframe(comp_df, use_container_width=True)

# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================

elif current_step == steps[3]:
    st.markdown('<h2 class="sub-header">Step 4: Model Training</h2>', unsafe_allow_html=True)
    
    if 'prepared_data' not in st.session_state:
        st.warning("Please prepare data first in Step 3!")
    else:
        data = st.session_state.prepared_data
        X_train, y_train = data['X_train'], data['y_train']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="step-box">
            <h4>🤖 Training Process:</h4>
            <p>The model learns by finding the best parameters that minimize the error 
            between predictions and actual values.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model training based on type
            if regression_type == "Simple Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                st.markdown("### 📐 Learned Equation:")
                st.markdown(f"**y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}**")
                
                # Compare with true values
                true_data = st.session_state.datasets[regression_type]['metadata']
                if 'true_slope' in true_data:
                    st.markdown("**Comparison with True Values:**")
                    st.write(f"Slope: Learned={model.coef_[0]:.3f}, True={true_data['true_slope']:.3f}")
                    st.write(f"Intercept: Learned={model.intercept_:.3f}, True={true_data['true_intercept']:.3f}")
            
            elif regression_type == "Multiple Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                st.markdown("### 📐 Learned Coefficients:")
                feature_names = st.session_state.datasets[regression_type]['metadata']['feature_names']
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': model.coef_
                })
                st.dataframe(coef_df, use_container_width=True)
                st.write(f"Intercept: {model.intercept_:.3f}")
            
            else:  # Polynomial Regression
                degree = st.slider("Select Polynomial Degree", 1, 10, 2)
                
                # Create polynomial features
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_train_poly = poly.fit_transform(X_train)
                
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                
                st.markdown(f"### 📊 Model trained with degree {degree}")
                st.write(f"Number of features after polynomial transform: {X_train_poly.shape[1]}")
                
                # Store polynomial transformer
                st.session_state.poly = poly
            
            # Store model
            st.session_state.model = model
            st.session_state.model_type = regression_type
            
            # Training metrics
            y_pred_train = model.predict(X_train if regression_type != "Polynomial Regression" 
                                         else X_train_poly)
            train_r2 = r2_score(y_train, y_pred_train)
            
            st.success(f"✅ Model trained successfully! Training R² Score: {train_r2:.3f}")
        
        with col2:
            st.markdown("### 📈 Training Progress Visualization")
            
            # Show learning curve
            if regression_type == "Simple Linear Regression" or regression_type == "Polynomial Regression":
                # For 1D problems, show the fitted line
                if X_train.shape[1] == 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=X_train.squeeze(), y=y_train, 
                                            mode='markers', name='Training Data',
                                            marker=dict(color='blue', size=8)))
                    
                    X_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
                    
                    if regression_type == "Polynomial Regression":
                        X_range_poly = poly.transform(X_range)
                        y_range = model.predict(X_range_poly)
                    else:
                        y_range = model.predict(X_range)
                    
                    fig.add_trace(go.Scatter(x=X_range.squeeze(), y=y_range,
                                            mode='lines', name='Fitted Line',
                                            line=dict(color='red', width=3)))
                    
                    fig.update_layout(title="Model Fit on Training Data",
                                    xaxis_title="X", yaxis_title="y")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance for multiple regression
            if regression_type == "Multiple Linear Regression":
                importance = np.abs(model.coef_)
                feature_names = st.session_state.datasets[regression_type]['metadata']['feature_names']
                
                fig = px.bar(x=feature_names, y=importance,
                           title="Feature Importance (Absolute Coefficient Values)",
                           labels={'x': 'Features', 'y': '|Coefficient|'})
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================

elif current_step == steps[4]:
    st.markdown('<h2 class="sub-header">Step 5: Model Evaluation</h2>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first in Step 4!")
    else:
        data = st.session_state.prepared_data
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
        model = st.session_state.model
        
        # Prepare test data (polynomial features if needed)
        if st.session_state.model_type == "Polynomial Regression" and 'poly' in st.session_state:
            X_test_poly = st.session_state.poly.transform(X_test)
            y_pred_test = model.predict(X_test_poly)
            X_train_poly = st.session_state.poly.transform(X_train)
            y_pred_train = model.predict(X_train_poly)
        else:
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("R² Score (Train)", f"{train_r2:.3f}")
            st.metric("R² Score (Test)", f"{test_r2:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("RMSE (Train)", f"{train_rmse:.3f}")
            st.metric("RMSE (Test)", f"{test_rmse:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("MAE (Train)", f"{train_mae:.3f}")
            st.metric("MAE (Test)", f"{test_mae:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### 📊 Evaluation Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residuals", "Learning Curve"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_train, y=y_pred_train, 
                                    mode='markers', name='Training',
                                    marker=dict(color='blue', size=8, opacity=0.6)))
            fig.add_trace(go.Scatter(x=y_test, y=y_pred_test,
                                    mode='markers', name='Testing',
                                    marker=dict(color='red', size=8, opacity=0.6)))
            
            # Perfect prediction line
            min_val = min(y_train.min(), y_test.min(), y_pred_train.min(), y_pred_test.min())
            max_val = max(y_train.max(), y_test.max(), y_pred_train.max(), y_pred_test.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                    mode='lines', name='Perfect Prediction',
                                    line=dict(color='green', dash='dash')))
            
            fig.update_layout(title="Actual vs Predicted Values",
                            xaxis_title="Actual Values",
                            yaxis_title="Predicted Values")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            residuals_train = y_train - y_pred_train
            residuals_test = y_test - y_pred_test
            
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=('Residuals vs Predicted', 'Residual Distribution'))
            
            fig.add_trace(go.Scatter(x=y_pred_train, y=residuals_train,
                                    mode='markers', name='Train',
                                    marker=dict(color='blue', opacity=0.6)), row=1, col=1)
            fig.add_trace(go.Scatter(x=y_pred_test, y=residuals_test,
                                    mode='markers', name='Test',
                                    marker=dict(color='red', opacity=0.6)), row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="green", row=1, col=1)
            
            fig.add_trace(go.Histogram(x=residuals_train, nbinsx=30, 
                                     name='Train', opacity=0.7), row=1, col=2)
            fig.add_trace(go.Histogram(x=residuals_test, nbinsx=30,
                                     name='Test', opacity=0.7), row=1, col=2)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Simplified learning curve
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_scores = []
            test_scores = []
            
            for size in train_sizes:
                n_train = int(len(X_train) * size)
                if n_train < 2:
                    continue
                
                X_subset = X_train[:n_train]
                y_subset = y_train[:n_train]
                
                if st.session_state.model_type == "Polynomial Regression" and 'poly' in st.session_state:
                    X_subset_poly = st.session_state.poly.transform(X_subset)
                    model_subset = LinearRegression()
                    model_subset.fit(X_subset_poly, y_subset)
                    train_score = model_subset.score(X_subset_poly, y_subset)
                    test_score = model_subset.score(X_test_poly, y_test)
                else:
                    model_subset = LinearRegression()
                    model_subset.fit(X_subset, y_subset)
                    train_score = model_subset.score(X_subset, y_subset)
                    test_score = model_subset.score(X_test, y_test)
                
                train_scores.append(train_score)
                test_scores.append(test_score)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_sizes[:len(train_scores)] * len(X_train),
                                    y=train_scores, mode='lines+markers',
                                    name='Training Score', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=train_sizes[:len(test_scores)] * len(X_train),
                                    y=test_scores, mode='lines+markers',
                                    name='Testing Score', line=dict(color='red')))
            
            fig.update_layout(title="Learning Curve",
                            xaxis_title="Training Examples",
                            yaxis_title="R² Score",
                            yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("""
        <div class="insight-box">
        <h4>📌 Model Interpretation:</h4>
        """, unsafe_allow_html=True)
        
        if abs(train_r2 - test_r2) < 0.1:
            st.markdown("✅ **Well-balanced model:** Training and testing performance are similar")
        elif train_r2 > test_r2 + 0.1:
            st.markdown("⚠️ **Possible overfitting:** Model performs much better on training data")
        else:
            st.markdown("📊 **Model might be underfitting:** Consider increasing model complexity")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================================

else:  # Step 6
    st.markdown('<h2 class="sub-header">Step 6: Make Predictions</h2>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first in Step 4!")
    else:
        st.markdown("""
        <div class="step-box">
        <h4>🔮 Make Predictions with Your Trained Model</h4>
        <p>Enter values below to see what your model predicts!</p>
        </div>
        """, unsafe_allow_html=True)
        
        model = st.session_state.model
        metadata = st.session_state.datasets[regression_type]['metadata']
        
        if regression_type == "Simple Linear Regression":
            x_input = st.number_input("Enter X value:", value=5.0, step=0.1)
            
            if st.button("Predict", type="primary"):
                X_input = np.array([[x_input]])
                prediction = model.predict(X_input)[0]
                
                st.markdown(f"""
                <div style='background-color: #1E88E5; padding: 2rem; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white;'>Predicted y = {prediction:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Show equation
                st.markdown(f"**Model Equation:** y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")
        
        elif regression_type == "Multiple Linear Regression":
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                x1 = st.number_input("Enter X₁ value:", value=5.0, step=0.1)
            with col_in2:
                x2 = st.number_input("Enter X₂ value:", value=5.0, step=0.1)
            with col_in3:
                x3 = st.number_input("Enter X₃ value:", value=5.0, step=0.1)
            
            if st.button("Predict", type="primary"):
                X_input = np.array([[x1, x2, x3]])
                
                # Scale if scaler exists
                if st.session_state.prepared_data['scaler']:
                    X_input = st.session_state.prepared_data['scaler'].transform(X_input)
                
                prediction = model.predict(X_input)[0]
                
                st.markdown(f"""
                <div style='background-color: #1E88E5; padding: 2rem; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white;'>Predicted y = {prediction:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Show equation
                equation = f"y = {model.intercept_:.3f}"
                for i, coef in enumerate(model.coef_):
                    equation += f" + {coef:.3f}×X{i+1}"
                st.markdown(f"**Model Equation:** {equation}")
        
        else:  # Polynomial Regression
            x_input = st.number_input("Enter X value:", value=5.0, step=0.1)
            
            if st.button("Predict", type="primary"):
                X_input = np.array([[x_input]])
                X_input_poly = st.session_state.poly.transform(X_input)
                prediction = model.predict(X_input_poly)[0]
                
                st.markdown(f"""
                <div style='background-color: #1E88E5; padding: 2rem; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white;'>Predicted y = {prediction:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# FOOTER WITH ADDITIONAL INFORMATION
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #f5f5f5; border-radius: 10px;'>
    <h4>📚 Learn More About Regression</h4>
    <p>
    <b>Simple Linear Regression:</b> Models relationship between two variables using a straight line<br>
    <b>Multiple Linear Regression:</b> Extends to multiple independent variables<br>
    <b>Polynomial Regression:</b> Captures non-linear relationships using polynomial terms
    </p>
    <p style='color: gray; font-size: 0.9rem;'>
    Built with Streamlit • Complete ML Pipeline • Interactive Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8rem;'>
        <p>Made with ❤️ for ML Learning</p>
        <p>Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="AutoML Pipeline Pro", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Aesthetics ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #e1f5fe; border-bottom: 2px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

# --- App Header ---
st.title("🚀 Advanced ML Pipeline Dashboard")
st.markdown("---")

# --- Step-based Navigation ---
tabs = st.tabs([
    "1. Setup & Data", "2. EDA", "3. Engineering", 
    "4. Selection", "5. Training", "6. Tuning"
])

# --- TAB 1: SETUP & DATA ---
with tabs[0]:
    st.header("Step 1: Data Configuration")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        problem_type = st.radio("Problem Type", ["Classification", "Regression"])
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        st.session_state.data = df
        
        with col2:
            target_col = st.selectbox("Select Target Feature", df.columns)
            features = st.multiselect("Select Input Features", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col])
            
        if len(features) >= 2:
            st.subheader("PCA Data Projection")
            X_pca = df[features].select_dtypes(include=[np.number]).dropna()
            if not X_pca.empty:
                pca = PCA(n_components=2)
                components = pca.fit_transform(StandardScaler().fit_transform(X_pca))
                fig_pca = px.scatter(x=components[:,0], y=components[:,1], color=df.loc[X_pca.index, target_col],
                                     title="PCA Projection (2D)", labels={'x': 'PC1', 'y': 'PC2'}, template="plotly_white")
                st.plotly_chart(fig_pca, use_container_width=True)

# --- TAB 2: EDA ---
with tabs[1]:
    if st.session_state.data is not None:
        st.header("Exploratory Data Analysis")
        df = st.session_state.data
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isna().sum().sum())
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Correlation Heatmap")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 3: DATA ENGINEERING & CLEANING ---
with tabs[2]:
    if st.session_state.data is not None:
        st.header("Cleaning & Outliers")
        df_clean = st.session_state.data.copy()
        
        st.subheader("1. Missing Value Imputation")
        method = st.selectbox("Imputation Method", ["Mean", "Median", "Mode"])
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if st.button("Apply Imputation"):
            strat = method.lower()
            imputer = SimpleImputer(strategy=strat)
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
            st.success("Imputation Applied")

        st.subheader("2. Outlier Detection")
        outlier_method = st.selectbox("Detection Method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        
        outliers = pd.Series(False, index=df_clean.index)
        
        if outlier_method == "IQR":
            Q1 = df_clean[num_cols].quantile(0.25)
            Q3 = df_clean[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df_clean[num_cols] < (Q1 - 1.5 * IQR)) | (df_clean[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        elif outlier_method == "Isolation Forest":
            iso = IsolationForest(contamination=0.05)
            outliers = iso.fit_predict(df_clean[num_cols]) == -1
            
        st.warning(f"Detected {sum(outliers)} outliers.")
        
        if sum(outliers) > 0:
            if st.button("Delete Selected Outliers"):
                df_clean = df_clean[~outliers]
                st.session_state.data = df_clean
                st.success("Outliers removed from session!")
                st.rerun()

# --- TAB 4: FEATURE SELECTION ---
with tabs[3]:
    if st.session_state.data is not None:
        st.header("Feature Selection")
        df_fs = st.session_state.data
        X = df_fs[features].select_dtypes(include=[np.number])
        y = df_fs[target_col]
        
        # Simple encoding for target if classification
        if problem_type == "Classification" and y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        sel_method = st.multiselect("Selection Criteria", ["Variance Threshold", "Information Gain / Mutual Info", "Correlation with Target"])
        
        if "Variance Threshold" in sel_method:
            vt = VarianceThreshold(threshold=0.1)
            vt.fit(X)
            st.write("Low Variance Features removed:", list(X.columns[~vt.get_support()]))
            
        if "Information Gain / Mutual Info" in sel_method:
            mi = mutual_info_classif(X, y) if problem_type == "Classification" else mutual_info_regression(X, y)
            mi_df = pd.DataFrame({'Feature': X.columns, 'Score': mi}).sort_values(by='Score', ascending=False)
            fig_mi = px.bar(mi_df, x='Feature', y='Score', title="Mutual Information Scores")
            st.plotly_chart(fig_mi)

# --- TAB 5: MODEL TRAINING & K-FOLD ---
with tabs[4]:
    if st.session_state.data is not None:
        st.header("Model Selection & Training")
        
        model_choice = st.selectbox("Select Model", ["Linear/Logistic Regression", "SVM", "Random Forest", "K-Means"])
        test_size = st.slider("Test Split Size", 0.1, 0.5, 0.2)
        k_val = st.number_input("K-Fold Value (K)", min_value=2, max_value=10, value=5)
        
        if st.button("Train & Validate"):
            X = st.session_state.data[features].select_dtypes(include=[np.number])
            y = st.session_state.data[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            
            # Model Mapping
            if model_choice == "Random Forest":
                model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
            elif model_choice == "SVM":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                model = SVC(kernel=kernel) if problem_type == "Classification" else SVR(kernel=kernel)
            else:
                model = LogisticRegression() if problem_type == "Classification" else LinearRegression()
            
            # K-Fold
            cv_results = cross_validate(model, X, y, cv=k_val, return_train_score=True)
            
            st.subheader("Performance Metrics")
            col_m1, col_m2 = st.columns(2)
            train_score = cv_results['train_score'].mean()
            test_score = cv_results['test_score'].mean()
            
            col_m1.metric("Avg Train Score", f"{train_score:.4f}")
            col_m2.metric("Avg Test Score", f"{test_score:.4f}")
            
            if train_score > test_score + 0.15:
                st.error("High Risk of Overfitting detected.")
            elif train_score < 0.5:
                st.warning("Model may be Underfitting.")
            else:
                st.success("Model appears well-balanced.")

# --- TAB 6: HYPERPARAMETER TUNING ---
with tabs[5]:
    st.header("Hyperparameter Tuning")
    search_type = st.radio("Search Method", ["Grid Search", "Random Search"])
    
    st.info("Example: Tuning Random Forest depth and estimators...")
    if st.button("Run Tuning"):
        # Placeholder for demonstration
        param_grid = {'n_estimators': [10, 50], 'max_depth': [None, 5, 10]}
        X = st.session_state.data[features].select_dtypes(include=[np.number])
        y = st.session_state.data[target_col]
        
        base_model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
        search = GridSearchCV(base_model, param_grid, cv=3) if search_type == "Grid Search" else RandomizedSearchCV(base_model, param_grid, cv=3)
        
        search.fit(X, y)
        st.write("Best Params:", search.best_params_)
        st.write("Best Score Improvements:", search.best_score_)
        

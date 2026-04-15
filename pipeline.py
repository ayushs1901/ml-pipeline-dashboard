import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression

# ------------------ CONFIG ------------------

st.set_page_config(page_title="AutoML Pipeline Pro", layout="wide")

# ------------------ DARK THEME ------------------

st.markdown("""

<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
[data-testid="stSidebar"] {
    background: #020617;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stButton>button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 10px;
    border: none;
}
button[data-baseweb="tab"] {
    background-color: #1e293b;
    color: white;
    border-radius: 8px;
}
</style>

""", unsafe_allow_html=True)

# ------------------ TITLE ------------------

st.markdown("""

<h1 style='text-align: center;'>🚀 AutoML Pipeline Pro</h1>
<h4 style='text-align: center;'>End-to-End Machine Learning Dashboard</h4>
""", unsafe_allow_html=True)

st.markdown("### 🔄 Pipeline: Setup → EDA → Engineering → Selection → Training → Tuning")
st.markdown("---")

# ------------------ SESSION ------------------

if "data" not in st.session_state:
st.session_state.data = None
if "features" not in st.session_state:
st.session_state.features = []
if "target" not in st.session_state:
st.session_state.target = None
if "problem" not in st.session_state:
st.session_state.problem = "Classification"

# ------------------ TABS ------------------

tabs = st.tabs(["Setup", "EDA", "Engineering", "Selection", "Training", "Tuning"])

# ==================================================

# SETUP

# ==================================================

with tabs[0]:
st.header("📂 Data Setup")

```
problem_type = st.radio("Problem Type", ["Classification", "Regression"])
st.session_state.problem = problem_type

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.session_state.data = df

    target = st.selectbox("Target Column", df.columns)
    features = st.multiselect(
        "Select Features",
        [c for c in df.columns if c != target],
        default=[c for c in df.columns if c != target]
    )

    st.session_state.target = target
    st.session_state.features = features

    st.dataframe(df.head(), use_container_width=True)

    if len(features) > 1:
        X = df[features].select_dtypes(include=np.number).dropna()
        if not X.empty:
            pca = PCA(n_components=2)
            comp = pca.fit_transform(StandardScaler().fit_transform(X))

            fig = px.scatter(
                x=comp[:, 0],
                y=comp[:, 1],
                color=df.loc[X.index, target],
                title="PCA Projection",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
```

# ==================================================

# EDA

# ==================================================

with tabs[1]:
if st.session_state.data is None:
st.warning("⚠️ Complete Setup first")
st.stop()

```
df = st.session_state.data

st.header("📊 EDA")

c1, c2, c3 = st.columns(3)
c1.metric("📊 Rows", df.shape[0])
c2.metric("📁 Columns", df.shape[1])
c3.metric("⚠️ Missing", df.isna().sum().sum())

st.dataframe(df.describe(), use_container_width=True)

missing = df.isnull().sum()
fig = px.bar(missing[missing > 0], title="Missing Values", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

corr = df.select_dtypes(include=np.number).corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='blues', template="plotly_dark")
st.plotly_chart(fig_corr)
```

# ==================================================

# ENGINEERING

# ==================================================

with tabs[2]:
if st.session_state.data is None:
st.stop()

```
df = st.session_state.data.copy()
num_cols = df.select_dtypes(include=np.number).columns

st.header("🛠 Cleaning")

method = st.selectbox("Imputation", ["Mean", "Median", "Mode"])

if st.button("Apply Imputation"):
    imputer = SimpleImputer(strategy=method.lower())
    df[num_cols] = imputer.fit_transform(df[num_cols])
    st.session_state.data = df
    st.success("Done")

method_o = st.selectbox("Outlier Method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])

if method_o == "Isolation Forest":
    iso = IsolationForest(contamination=0.05)
    outliers = iso.fit_predict(df[num_cols]) == -1
    st.write("Outliers:", sum(outliers))
```

# ==================================================

# SELECTION

# ==================================================

with tabs[3]:
if st.session_state.data is None:
st.stop()

```
df = st.session_state.data
features = st.session_state.features
target = st.session_state.target

st.header("🎯 Feature Selection")

X = df[features].select_dtypes(include=np.number)
y = df[target]

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

if st.checkbox("Variance Threshold"):
    vt = VarianceThreshold()
    vt.fit(X)
    st.write("Removed:", list(X.columns[~vt.get_support()]))

if st.checkbox("Mutual Info"):
    mi = mutual_info_classif(X, y) if st.session_state.problem == "Classification" else mutual_info_regression(X, y)
    mi_df = pd.DataFrame({"Feature": X.columns, "Score": mi})
    fig = px.bar(mi_df, x="Feature", y="Score", template="plotly_dark")
    st.plotly_chart(fig)
```

# ==================================================

# TRAINING

# ==================================================

with tabs[4]:
if st.session_state.data is None:
st.stop()

```
df = st.session_state.data
X = df[st.session_state.features].select_dtypes(include=np.number)
y = df[st.session_state.target]

st.header("🤖 Training")

model_name = st.selectbox("Model", ["Linear", "SVM", "Random Forest"])

if model_name == "Random Forest":
    model = RandomForestClassifier() if st.session_state.problem == "Classification" else RandomForestRegressor()
elif model_name == "SVM":
    model = SVC() if st.session_state.problem == "Classification" else SVR()
else:
    model = LogisticRegression() if st.session_state.problem == "Classification" else LinearRegression()

if st.button("Train Model"):
    cv = cross_validate(model, X, y, cv=5, return_train_score=True)

    fig = go.Figure()
    fig.add_bar(name="Train", x=["Score"], y=[np.mean(cv["train_score"])])
    fig.add_bar(name="Test", x=["Score"], y=[np.mean(cv["test_score"])])
    fig.update_layout(barmode='group', template="plotly_dark")

    st.plotly_chart(fig)
```

# ==================================================

# TUNING

# ==================================================

with tabs[5]:
if st.session_state.data is None:
st.stop()

```
df = st.session_state.data
X = df[st.session_state.features].select_dtypes(include=np.number)
y = df[st.session_state.target]

st.header("⚙️ Tuning")

if st.button("Run Grid Search"):
    param = {"n_estimators": [50, 100], "max_depth": [None, 5]}
    model = RandomForestClassifier() if st.session_state.problem == "Classification" else RandomForestRegressor()

    grid = GridSearchCV(model, param, cv=3)
    grid.fit(X, y)

    st.write("Best Params:", grid.best_params_)
    st.write("Best Score:", grid.best_score_)
```

st.success("✅ App Running Successfully")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ----------------

st.set_page_config(page_title="AutoML", layout="wide")

# ---------------- DARK THEME ----------------

st.markdown("""

<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
</style>

""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown("<h1 style='text-align:center;'>🚀 AutoML Dashboard</h1>", unsafe_allow_html=True)

# ---------------- SESSION ----------------

if "data" not in st.session_state:
st.session_state.data = None

# ---------------- FILE UPLOAD ----------------

file = st.file_uploader("Upload CSV")

if file is not None:
df = pd.read_csv(file)
st.session_state.data = df

# ---------------- IF DATA EXISTS ----------------

if st.session_state.data is not None:

```
df = st.session_state.data

st.write("### Data Preview")
st.dataframe(df.head())

target = st.selectbox("Select Target", df.columns)

features = [c for c in df.columns if c != target]

X = df[features].select_dtypes(include=np.number)
y = df[target]

if st.button("Train Model"):
    model = RandomForestClassifier()
    scores = cross_val_score(model, X, y, cv=5)

    st.write("Accuracy:", scores.mean())
```

else:
st.warning("Upload dataset first")

import json, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = os.environ.get("OUT_DIR", "output")
st.set_page_config(page_title="Car Insurance Claim Dashboard", page_icon="", layout="wide")

# ==========================================
# UI/UX: ENTERPRISE CLEAN DASHBOARD
# ==========================================
st.markdown("""
    <style>
    /* 1. FONT PROFESIONAL */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* 2. BACKGROUND BERSIH */
    .stApp {
        background-color: #F8FAFC !important; 
        font-family: 'Inter', sans-serif !important;
    }
    
    /* 3. SEMBUNYIIN BAWAAN STREAMLIT */
    #MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton {display: none !important;}
    .block-container { padding-top: 1.5rem !important; max-width: 95% !important; }

    /* 4. HERO BANNER (Hiasan di atas) */
    .hero-banner {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .hero-logo {
        background-color: #2563EB;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 24px;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        letter-spacing: -1px;
    }
    .hero-text h1 {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 1.6rem !important;
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    .hero-text p {
        margin: 0 !important;
        padding: 0 !important;
        color: #64748B !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    .hero-badge {
        margin-left: auto;
        background-color: #EFF6FF;
        color: #2563EB;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* 5. METRIC CARDS (Interaktif) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 4px solid #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 20px !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important; 
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        border-left: 4px solid #2563EB !important;
    }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #0F172A !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; color: #64748B !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px;}

    /* 6. TAB MENU */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #E2E8F0; }
    .stTabs [data-baseweb="tab"] { padding: 10px 0px !important; font-weight: 600 !important; color: #64748B !important; background-color: transparent !important; transition: color 0.2s ease; }
    .stTabs [data-baseweb="tab"]:hover { color: #2563EB !important; }
    .stTabs [aria-selected="true"] { color: #2563EB !important; border-bottom: 3px solid #2563EB !important; }

    /* 7. INFO BOX & LAINNYA */
    .stAlert { border-radius: 8px !important; border: 1px solid #E2E8F0 !important; background-color: #F8FAFC !important; color: #334155 !important; }
    hr { border-color: #E2E8F0 !important; margin: 1.5rem 0 !important; }
    </style>
""", unsafe_allow_html=True)

# Fungsi standar grafik profesional
def pro_chart_style(fig, height=400):
    fig.update_layout(
        height=height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=10, r=10),
        title_font=dict(size=15, color="#1E293B", family="Inter", weight="bold"),
        font=dict(family="Inter, sans-serif", color="#64748B", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None)
    )
    fig.update_xaxes(showgrid=False, linecolor='#E2E8F0')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9', linecolor='rgba(0,0,0,0)')
    return fig

PLOTLY_CONFIG = {'displayModeBar': False}

@st.cache_data
def load():
    d = dict(
        clean=pd.read_csv(f"{OUT}/car_insurance_clean.csv"),
        metrics=pd.read_csv(f"{OUT}/metrics.csv"),
        roc=pd.read_csv(f"{OUT}/roc_curve.csv"),
        fi=pd.read_csv(f"{OUT}/feature_importance.csv"),
        eda=pd.read_csv(f"{OUT}/eda_categorical.csv"),
        corr=pd.read_csv(f"{OUT}/correlation.csv", index_col=0),
        missing=pd.read_csv(f"{OUT}/missing.csv", index_col=0),
        summary=json.load(open(f"{OUT}/summary.json")),
    )
    return d

D = load()
S = D["summary"]
M = D["metrics"]

# ==========================================
# HEADER & HERO SECTION
# ==========================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-logo">CI</div>
    <div class="hero-text">
        <h1>Car Insurance Claim Prediction</h1>
        <p>Logistic Regression vs Random Forest using PySpark — Kaggle Dataset</p>
    </div>
    <div class="hero-badge">Enterprise Dashboard</div>
</div>
""", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan Data", "Eksplorasi (EDA)", "Perbandingan Model", "Data & Basis Data"])

# ---------------- Tab 1: RINGKASAN
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    claim = S["target"]["1"]
    c1.metric("Total Data", f"{S['n_rows']:,}")
    c2.metric("Jumlah Variabel", S["n_cols"])
    c3.metric("Tingkat Klaim", f"{claim / S['n_rows'] * 100:.1f}%")
    c4.metric("Model Terbaik (ROC-AUC)", S["best_model"])

    st.write("")
    st.markdown("---")

    l, r = st.columns([1, 1.5])
    
    tdf = pd.DataFrame({"Status": ["Tidak klaim", "Klaim"], "Jumlah": [S["target"]["0"], S["target"]["1"]]})
    fig_pie = px.pie(tdf, names="Status", values="Jumlah", hole=0.6, 
                     title="Distribusi Target (OUTCOME)",
                     color_discrete_sequence=["#94A3B8", "#EF4444"])
    fig_pie.update_traces(textposition='inside', textinfo='percent', marker=dict(line=dict(color='#F8FAFC', width=2)))
    fig_pie = pro_chart_style(fig_pie, height=350)
    fig_pie.update_layout(showlegend=True)
    l.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)

    miss = D["missing"][D["missing"]["missing"] > 0].reset_index().rename(columns={"index": "variabel"})
    fig_bar = px.bar(miss, x="variabel", y="pct", text="pct", 
                     title="Persentase Missing Value",
                     color_discrete_sequence=["#3B82F6"])
    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_bar = pro_chart_style(fig_bar, height=350)
    fig_bar.update_layout(yaxis_title="Persentase (%)", xaxis_title="")
    r.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    st.info(f"Data Cleaning Info: Duplikat baris: {S['n_dup']} | Data Train: {S['train_n']:,} | Data Test: {S['test_n']:,}")

# ---------------- Tab 2: EDA
with tab2:
    eda = D["eda"]
    var = st.selectbox("Pilih variabel kategorikal", eda["variabel"].unique(), index=0)
    sub = eda[eda["variabel"] == var]
    a, b = st.columns(2)
    
    fig_eda1 = px.bar(sub, x="kategori", y="jumlah", title=f"Jumlah Pemegang Polis per {var}",
                      color_discrete_sequence=["#3B82F6"])
    fig_eda1 = pro_chart_style(fig_eda1, height=350)
    a.plotly_chart(fig_eda1, use_container_width=True, config=PLOTLY_CONFIG)
    
    fig_eda2 = px.bar(sub, x="kategori", y="claim_rate", text="claim_rate",
                      title=f"Tingkat Klaim (%) per {var}", color_discrete_sequence=["#EF4444"])
    fig_eda2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_eda2 = pro_chart_style(fig_eda2, height=350)
    b.plotly_chart(fig_eda2, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    cl = D["clean"]
    num = st.selectbox("Distribusi variabel numerik", ["CREDIT_SCORE", "ANNUAL_MILEAGE", "SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS"])
    cl2 = cl.assign(Status=cl["OUTCOME"].map({0: "Tidak klaim", 1: "Klaim"}))
    
    fig_hist = px.histogram(cl2, x=num, color="Status", barmode="overlay", opacity=0.8,
                            title=f"Distribusi {num} berdasarkan status klaim",
                            color_discrete_map={"Tidak klaim": "#94A3B8", "Klaim": "#EF4444"})
    fig_hist = pro_chart_style(fig_hist, height=400)
    st.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CONFIG)
    
    fig_corr = px.imshow(D["corr"], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Heatmap Korelasi", aspect="auto")
    fig_corr = pro_chart_style(fig_corr, height=500)
    st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------- Tab 3: MODEL
with tab3:
    st.subheader("Evaluasi Performa Model")
    show = M[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns[1:]}).highlight_max(
        subset=list(show.columns[1:]), color="#E2E8F0"), use_container_width=True, hide_index=True)

    long = show.melt(id_vars="model", var_name="metrik", value_name="nilai")
    fig_metric = px.bar(long, x="metrik", y="nilai", color="model", barmode="group", text_auto=".3f",
                        title="Perbandingan Metrik Evaluasi", range_y=[0, 1.15],
                        color_discrete_sequence=["#3B82F6", "#10B981"])
    fig_metric = pro_chart_style(fig_metric, height=400)
    st.plotly_chart(fig_metric, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    a, b = st.columns(2)
    
    fig_roc = go.Figure()
    for m, g in D["roc"].groupby("model"):
        auc = M.loc[M.model == m, "roc_auc"].iloc[0]
        fig_roc.add_trace(go.Scatter(x=g["fpr"], y=g["tpr"], mode="lines", name=f"{m} (AUC={auc:.3f})", line=dict(width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94A3B8"), name="Acak"))
    fig_roc.update_layout(title="Kurva ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    fig_roc = pro_chart_style(fig_roc, height=400)
    a.plotly_chart(fig_roc, use_container_width=True, config=PLOTLY_CONFIG)

    mdl = b.radio("Pilih model untuk Confusion Matrix:", M["model"].tolist(), horizontal=True)
    row = M[M.model == mdl].iloc[0]
    cm = pd.DataFrame([[row.tn, row.fp], [row.fn, row.tp]], index=["Aktual: Tidak", "Aktual: Klaim"],
                      columns=["Prediksi: Tidak", "Prediksi: Klaim"])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=f"Confusion Matrix — {mdl}")
    fig_cm = pro_chart_style(fig_cm, height=400)
    fig_cm.update_layout(coloraxis_showscale=False)
    b.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    st.subheader("Feature Importance")
    fm = st.radio("Pilih Model:", ["Random Forest", "Logistic Regression"], horizontal=True, key="fi")
    f = D["fi"][D["fi"].model == fm].sort_values("abs", ascending=False).head(10).iloc[::-1]
    lab = "Importance" if fm == "Random Forest" else "Koefisien"
    
    fig_fi = px.bar(f, x="nilai", y="fitur", orientation="h", title=f"Top 10 Fitur — {fm}",
                    labels={"nilai": lab}, color_discrete_sequence=["#3B82F6"])
    fig_fi = pro_chart_style(fig_fi, height=400)
    st.plotly_chart(fig_fi, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(f"Hyperparameter terbaik (3-fold CV) — LR: {S['lr_params']} • RF: {S['rf_params']}")

# ---------------- Tab 4: DATA
with tab4:
    st.subheader("Dataset Hasil Olahan")
    st.dataframe(D["clean"].head(200), use_container_width=True)
    st.download_button("Unduh dataset olahan (CSV)", D["clean"].to_csv(index=False).encode(),
                       "car_insurance_clean.csv", "text/csv")
    st.markdown("---")
    st.markdown("""
    **Keterkaitan dengan basis data**
    - **Penyimpanan:** data polis (RDBMS/data lake) dibaca Spark, lalu disimpan ulang sebagai *Parquet* (columnar, terkompresi, cepat untuk query analitik).
    - **Pengambilan data:** dengan `createOrReplaceTempView` data dapat di-query menggunakan **Spark SQL**, mis. `SELECT AGE, AVG(OUTCOME) FROM claims GROUP BY AGE`.
    - **Integrasi sistem:** hasil prediksi dapat ditulis kembali ke basis data (PostgreSQL/MySQL via JDBC) untuk dipakai sistem underwriting & dashboard.
    """)

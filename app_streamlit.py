import json, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = os.environ.get("OUT_DIR", "output")
st.set_page_config(page_title="Prediksi Klaim Asuransi Kendaraan", page_icon="🚗", layout="wide")

# ==========================================
# UI/UX LEVEL DEWA: GLASSMORPHISM & MESH GRADIENT
# ==========================================
st.markdown("""
    <style>
    /* 1. IMPORT FONT (Plus Jakarta Sans & Space Grotesk buat Angka) */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
    
    /* 2. BACKGROUND MESH GRADIENT (Gak Flat!) */
    .stApp {
        background-color: #F4F7FB !important;
        background-image: 
            radial-gradient(at 0% 0%, hsla(210,100%,93%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(0,100%,93%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(240,100%,93%,1) 0, transparent 50%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* 3. Sembunyiin elemen bawaan Streamlit */
    #MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton {display: none !important;}
    .block-container { padding-top: 2rem !important; max-width: 1200px !important; }

    /* 4. Tipografi */
    h1 {
        font-weight: 700 !important;
        color: #0F172A !important;
        letter-spacing: -0.05rem;
        font-size: 2.5rem !important;
    }
    .stCaption {
        color: #475569 !important;
        font-size: 1.05rem !important;
    }

    /* 5. GLASSMORPHISM CARD UNTUK METRIK */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.15) !important;
    }
    
    /* Font Angka (Techy Look) */
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2rem !important; 
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #64748B !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }

    /* 6. Glassmorphism untuk Tab Menu */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.4);
        padding: 8px;
        border-radius: 12px;
        backdrop-filter: blur(5px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #FF4B4B !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
    }
    
    /* 7. Info Box */
    .stAlert { 
        border-radius: 12px !important; 
        border: 1px solid #BFDBFE !important;
        background: rgba(239, 246, 255, 0.8) !important; 
        color: #1E40AF !important; 
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi sakti biar grafiknya transparan & rapi
def premium_chart_style(fig, height=400):
    fig.update_layout(
        height=height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=30, l=20, r=20),
        title_font=dict(size=16, color="#0F172A", family="Plus Jakarta Sans"),
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#475569"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False, linecolor='rgba(200,200,200,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)', linecolor='rgba(0,0,0,0)')
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

# HEADER
st.title("🚗 Klaim Asuransi: LR vs RF")
st.caption("Analisis Prediktif menggunakan PySpark • Dataset: Car Insurance Claim (Kaggle)")
st.write("") 

tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan", "🔎 Eksplorasi (EDA)", "🤖 Model", "🗄️ Data & Basis Data"])

# ---------------- Tab 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    claim = S["target"]["1"]
    c1.metric("Jumlah Data", f"{S['n_rows']:,}")
    c2.metric("Jumlah Variabel", S["n_cols"])
    c3.metric("Tingkat Klaim", f"{claim / S['n_rows'] * 100:.1f}%")
    c4.metric("Model Terbaik", S["best_model"])

    st.write("---")

    l, r = st.columns([1, 1.5])
    
    tdf = pd.DataFrame({"Status": ["Tidak klaim", "Klaim"], "Jumlah": [S["target"]["0"], S["target"]["1"]]})
    fig_pie = px.pie(tdf, names="Status", values="Jumlah", hole=0.65, 
                     title="Distribusi Target (OUTCOME)",
                     color_discrete_sequence=["#CBD5E1", "#FF4B4B"])
    fig_pie.update_traces(textposition='inside', textinfo='percent', marker=dict(line=dict(color='#F4F7FB', width=3)))
    fig_pie = premium_chart_style(fig_pie, height=380)
    fig_pie.update_layout(showlegend=False)
    l.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)

    miss = D["missing"][D["missing"]["missing"] > 0].reset_index().rename(columns={"index": "variabel"})
    fig_bar = px.bar(miss, x="variabel", y="pct", text="pct", 
                     title="Persentase Missing Value (%)",
                     color_discrete_sequence=["#F59E0B"])
    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside', width=0.4, marker=dict(cornerradius=8))
    fig_bar = premium_chart_style(fig_bar, height=380)
    fig_bar.update_layout(yaxis_title="", xaxis_title="")
    r.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    st.info(f"**Info Sistem:** Duplikat baris: {S['n_dup']} | Data Train: {S['train_n']:,} | Data Test: {S['test_n']:,}")

# ---------------- Tab 2
with tab2:
    eda = D["eda"]
    var = st.selectbox("Pilih variabel kategorikal", eda["variabel"].unique(), index=0)
    sub = eda[eda["variabel"] == var]
    a, b = st.columns(2)
    
    fig_eda1 = px.bar(sub, x="kategori", y="jumlah", title=f"Jumlah Pemegang Polis per {var}",
                      color_discrete_sequence=["#3B82F6"])
    fig_eda1.update_traces(marker=dict(cornerradius=6))
    fig_eda1 = premium_chart_style(fig_eda1, height=350)
    a.plotly_chart(fig_eda1, use_container_width=True, config=PLOTLY_CONFIG)
    
    fig_eda2 = px.bar(sub, x="kategori", y="claim_rate", text="claim_rate",
                      title=f"Tingkat Klaim (%) per {var}", color_discrete_sequence=["#FF4B4B"])
    fig_eda2.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker=dict(cornerradius=6))
    fig_eda2 = premium_chart_style(fig_eda2, height=350)
    b.plotly_chart(fig_eda2, use_container_width=True, config=PLOTLY_CONFIG)

    st.write("---")
    cl = D["clean"]
    num = st.selectbox("Distribusi variabel numerik", ["CREDIT_SCORE", "ANNUAL_MILEAGE", "SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS"])
    cl2 = cl.assign(Status=cl["OUTCOME"].map({0: "Tidak klaim", 1: "Klaim"}))
    
    fig_hist = px.histogram(cl2, x=num, color="Status", barmode="overlay", opacity=0.8,
                            title=f"Distribusi {num} berdasarkan status klaim",
                            color_discrete_map={"Tidak klaim": "#CBD5E1", "Klaim": "#FF4B4B"})
    fig_hist = premium_chart_style(fig_hist, height=400)
    st.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CONFIG)
    
    fig_corr = px.imshow(D["corr"], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Heatmap Korelasi", aspect="auto")
    fig_corr = premium_chart_style(fig_corr, height=500)
    st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------- Tab 3
with tab3:
    st.subheader("Performa Model pada Data Uji")
    show = M[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns[1:]}).highlight_max(
        subset=list(show.columns[1:]), color="#bbf7d0"), use_container_width=True, hide_index=True)

    long = show.melt(id_vars="model", var_name="metrik", value_name="nilai")
    fig_metric = px.bar(long, x="metrik", y="nilai", color="model", barmode="group", text_auto=".3f",
                        title="Perbandingan Metrik Evaluasi", range_y=[0, 1.15],
                        color_discrete_sequence=["#3B82F6", "#10B981"])
    fig_metric.update_traces(marker=dict(cornerradius=6))
    fig_metric = premium_chart_style(fig_metric, height=400)
    st.plotly_chart(fig_metric, use_container_width=True, config=PLOTLY_CONFIG)

    st.write("---")
    a, b = st.columns(2)
    
    fig_roc = go.Figure()
    for m, g in D["roc"].groupby("model"):
        auc = M.loc[M.model == m, "roc_auc"].iloc[0]
        fig_roc.add_trace(go.Scatter(x=g["fpr"], y=g["tpr"], mode="lines", name=f"{m} (AUC={auc:.3f})", line=dict(width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Acak"))
    fig_roc.update_layout(title="Kurva ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    fig_roc = premium_chart_style(fig_roc, height=400)
    a.plotly_chart(fig_roc, use_container_width=True, config=PLOTLY_CONFIG)

    mdl = b.radio("Pilih model untuk Confusion Matrix:", M["model"].tolist(), horizontal=True)
    row = M[M.model == mdl].iloc[0]
    cm = pd.DataFrame([[row.tn, row.fp], [row.fn, row.tp]], index=["Aktual: Tidak", "Aktual: Klaim"],
                      columns=["Prediksi: Tidak", "Prediksi: Klaim"])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=f"Confusion Matrix – {mdl}")
    fig_cm = premium_chart_style(fig_cm, height=400)
    fig_cm.update_layout(coloraxis_showscale=False)
    b.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)

    st.write("---")
    st.subheader("Variabel Paling Berpengaruh")
    fm = st.radio("Pilih Model untuk Feature Importance:", ["Random Forest", "Logistic Regression"], horizontal=True, key="fi")
    f = D["fi"][D["fi"].model == fm].sort_values("abs", ascending=False).head(10).iloc[::-1]
    lab = "Importance" if fm == "Random Forest" else "Koefisien"
    
    fig_fi = px.bar(f, x="nilai", y="fitur", orientation="h", title=f"Top 10 Fitur – {fm}",
                    labels={"nilai": lab}, color_discrete_sequence=["#8B5CF6"])
    fig_fi.update_traces(marker=dict(cornerradius=6))
    fig_fi = premium_chart_style(fig_fi, height=400)
    st.plotly_chart(fig_fi, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(f"Hyperparameter terbaik (3-fold CV) → LR: {S['lr_params']} • RF: {S['rf_params']}")

# ---------------- Tab 4
with tab4:
    st.subheader("Dataset hasil olahan")
    st.dataframe(D["clean"].head(200), use_container_width=True)
    st.download_button("⬇️ Unduh dataset olahan (CSV)", D["clean"].to_csv(index=False).encode(),
                       "car_insurance_clean.csv", "text/csv")
    st.markdown("""
    **Keterkaitan dengan basis data**
    - **Penyimpanan:** data polis (RDBMS/data lake) → dibaca Spark → disimpan ulang sebagai *Parquet* (columnar, terkompresi, cepat untuk query analitik).
    - **Pengambilan data:** dengan `createOrReplaceTempView` data dapat di-query menggunakan **Spark SQL**, mis. `SELECT AGE, AVG(OUTCOME) FROM claims GROUP BY AGE`.
    - **Integrasi sistem:** hasil prediksi dapat ditulis kembali ke basis data (PostgreSQL/MySQL via JDBC) untuk dipakai sistem underwriting & dashboard.
    """)

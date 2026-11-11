import json, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = os.environ.get("OUT_DIR", "output")
# Ganti title & icon jadi lebih formal
st.set_page_config(page_title="Prediksi Klaim Asuransi Kendaraan", page_icon="", layout="wide")

# ==========================================
# UI/UX: CLEAN CORPORATE & PROFESSIONAL
# ==========================================
st.markdown("""
    <style>
    /* 1. FONT PROFESIONAL (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* 2. BACKGROUND BERSIH (Putih Tulang / Off-White) */
    .stApp {
        background-color: #F7F9FC !important; /* Off-white yang sangat soft, standar corporate */
        font-family: 'Inter', sans-serif !important;
    }

    /* 3. Sembunyiin elemen bawaan Streamlit */
    #MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton {display: none !important;}
    .block-container { padding-top: 2rem !important; max-width: 1200px !important; }

    /* 4. Tipografi Minimalis */
    h1 {
        font-weight: 700 !important;
        color: #111827 !important; /* Dark gray, bukan hitam pekat */
        letter-spacing: -0.5px;
        font-size: 2.2rem !important;
    }
    .stCaption {
        color: #6B7280 !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
    }
    
    h2, h3 {
        font-weight: 600 !important;
        color: #374151 !important;
    }

    /* 5. METRIC CARDS (Kotak Putih Bersih dengan Border Tipis) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important; /* Border abu tipis */
        border-radius: 8px !important; /* Sudut gak terlalu bulat */
        padding: 16px 20px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important; /* Shadow sangat halus */
    }
    
    /* Angka Metric */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; 
        color: #111827 !important;
        font-weight: 600 !important;
    }
    /* Label Metric */
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #6B7280 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* 6. TAB MENU (Bersih & Standar) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #E5E7EB; /* Garis pembatas bawah */
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 0px !important;
        font-weight: 500 !important;
        color: #6B7280 !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important; /* Biru corporate */
        border-bottom: 2px solid #2563EB !important;
    }
    
    /* 7. INFO BOX (Netral) */
    .stAlert { 
        border-radius: 6px !important; 
        border: 1px solid #E5E7EB !important;
        background-color: #FFFFFF !important; 
        color: #374151 !important; 
    }
    
    /* 8. Garis Pemisah (Divider) */
    hr { margin-top: 1rem; margin-bottom: 1rem; border-color: #E5E7EB; }
    </style>
""", unsafe_allow_html=True)

# Fungsi standar grafik profesional
def clean_chart_style(fig, height=400):
    fig.update_layout(
        height=height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=20, l=10, r=10),
        title_font=dict(size=14, color="#374151", family="Inter", weight="bold"),
        font=dict(family="Inter, sans-serif", color="#6B7280", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None)
    )
    fig.update_xaxes(showgrid=False, linecolor='#E5E7EB')
    fig.update_yaxes(showgrid=True, gridcolor='#F3F4F6', linecolor='rgba(0,0,0,0)')
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

# HEADER (Tanpa Emoji)
st.title("Logistic Regression vs Random Forest")
st.caption("Prediksi klaim asuransi kendaraan bermotor menggunakan PySpark — Dataset: Car Insurance Claim (Kaggle)")
st.write("") 

# TABS (Tanpa Emoji)
tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan Data", "Eksplorasi (EDA)", "Perbandingan Model", "Data & Basis Data"])

# ---------------- Tab 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    claim = S["target"]["1"]
    c1.metric("Jumlah Data", f"{S['n_rows']:,}")
    c2.metric("Jumlah Variabel", S["n_cols"])
    c3.metric("Tingkat Klaim", f"{claim / S['n_rows'] * 100:.1f}%")
    c4.metric("Model Terbaik (ROC-AUC)", S["best_model"])

    st.write("")
    st.markdown("---")

    l, r = st.columns([1, 1.5])
    
    tdf = pd.DataFrame({"Status": ["Tidak klaim", "Klaim"], "Jumlah": [S["target"]["0"], S["target"]["1"]]})
    fig_pie = px.pie(tdf, names="Status", values="Jumlah", hole=0.6, 
                     title="Distribusi Target (OUTCOME)",
                     color_discrete_sequence=["#9CA3AF", "#EF4444"]) # Abu netral & Merah tegas
    fig_pie.update_traces(textposition='inside', textinfo='percent', marker=dict(line=dict(color='#F7F9FC', width=2)))
    fig_pie = clean_chart_style(fig_pie, height=350)
    fig_pie.update_layout(showlegend=True)
    l.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)

    miss = D["missing"][D["missing"]["missing"] > 0].reset_index().rename(columns={"index": "variabel"})
    fig_bar = px.bar(miss, x="variabel", y="pct", text="pct", 
                     title="Persentase Missing Value",
                     color_discrete_sequence=["#3B82F6"]) # Biru corporate
    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_bar = clean_chart_style(fig_bar, height=350)
    fig_bar.update_layout(yaxis_title="Persentase (%)", xaxis_title="")
    r.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    st.info(f"Duplikat baris: {S['n_dup']}  |  Data Train: {S['train_n']:,}  |  Data Test: {S['test_n']:,}")

# ---------------- Tab 2
with tab2:
    eda = D["eda"]
    var = st.selectbox("Pilih variabel kategorikal", eda["variabel"].unique(), index=0)
    sub = eda[eda["variabel"] == var]
    a, b = st.columns(2)
    
    fig_eda1 = px.bar(sub, x="kategori", y="jumlah", title=f"Jumlah Pemegang Polis per {var}",
                      color_discrete_sequence=["#3B82F6"])
    fig_eda1 = clean_chart_style(fig_eda1, height=350)
    a.plotly_chart(fig_eda1, use_container_width=True, config=PLOTLY_CONFIG)
    
    fig_eda2 = px.bar(sub, x="kategori", y="claim_rate", text="claim_rate",
                      title=f"Tingkat Klaim (%) per {var}", color_discrete_sequence=["#EF4444"])
    fig_eda2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_eda2 = clean_chart_style(fig_eda2, height=350)
    b.plotly_chart(fig_eda2, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    cl = D["clean"]
    num = st.selectbox("Distribusi variabel numerik", ["CREDIT_SCORE", "ANNUAL_MILEAGE", "SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS"])
    cl2 = cl.assign(Status=cl["OUTCOME"].map({0: "Tidak klaim", 1: "Klaim"}))
    
    fig_hist = px.histogram(cl2, x=num, color="Status", barmode="overlay", opacity=0.8,
                            title=f"Distribusi {num} berdasarkan status klaim",
                            color_discrete_map={"Tidak klaim": "#9CA3AF", "Klaim": "#EF4444"})
    fig_hist = clean_chart_style(fig_hist, height=400)
    st.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CONFIG)
    
    fig_corr = px.imshow(D["corr"], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Heatmap Korelasi", aspect="auto")
    fig_corr = clean_chart_style(fig_corr, height=500)
    st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------- Tab 3
with tab3:
    st.subheader("Metrik Evaluasi pada Data Uji")
    show = M[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns[1:]}).highlight_max(
        subset=list(show.columns[1:]), color="#E5E7EB"), use_container_width=True, hide_index=True)

    long = show.melt(id_vars="model", var_name="metrik", value_name="nilai")
    fig_metric = px.bar(long, x="metrik", y="nilai", color="model", barmode="group", text_auto=".3f",
                        title="Perbandingan Metrik Evaluasi", range_y=[0, 1.15],
                        color_discrete_sequence=["#3B82F6", "#10B981"])
    fig_metric = clean_chart_style(fig_metric, height=400)
    st.plotly_chart(fig_metric, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    a, b = st.columns(2)
    
    fig_roc = go.Figure()
    for m, g in D["roc"].groupby("model"):
        auc = M.loc[M.model == m, "roc_auc"].iloc[0]
        fig_roc.add_trace(go.Scatter(x=g["fpr"], y=g["tpr"], mode="lines", name=f"{m} (AUC={auc:.3f})", line=dict(width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#9CA3AF"), name="Acak"))
    fig_roc.update_layout(title="Kurva ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    fig_roc = clean_chart_style(fig_roc, height=400)
    a.plotly_chart(fig_roc, use_container_width=True, config=PLOTLY_CONFIG)

    mdl = b.radio("Pilih model untuk Confusion Matrix:", M["model"].tolist(), horizontal=True)
    row = M[M.model == mdl].iloc[0]
    cm = pd.DataFrame([[row.tn, row.fp], [row.fn, row.tp]], index=["Aktual: Tidak", "Aktual: Klaim"],
                      columns=["Prediksi: Tidak", "Prediksi: Klaim"])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=f"Confusion Matrix — {mdl}")
    fig_cm = clean_chart_style(fig_cm, height=400)
    fig_cm.update_layout(coloraxis_showscale=False)
    b.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    st.subheader("Variabel Paling Berpengaruh")
    fm = st.radio("Pilih Model untuk Feature Importance:", ["Random Forest", "Logistic Regression"], horizontal=True, key="fi")
    f = D["fi"][D["fi"].model == fm].sort_values("abs", ascending=False).head(10).iloc[::-1]
    lab = "Importance" if fm == "Random Forest" else "Koefisien"
    
    fig_fi = px.bar(f, x="nilai", y="fitur", orientation="h", title=f"Top 10 Fitur — {fm}",
                    labels={"nilai": lab}, color_discrete_sequence=["#3B82F6"])
    fig_fi = clean_chart_style(fig_fi, height=400)
    st.plotly_chart(fig_fi, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(f"Hyperparameter terbaik (3-fold CV) — LR: {S['lr_params']} • RF: {S['rf_params']}")

# ---------------- Tab 4
with tab4:
    st.subheader("Dataset Hasil Olahan")
    st.dataframe(D["clean"].head(200), use_container_width=True)
    st.download_button("Unduh dataset olahan (CSV)", D["clean"].to_csv(index=False).encode(),
                       "car_insurance_clean.csv", "text/csv")
    st.markdown("""
    **Keterkaitan dengan basis data**
    - **Penyimpanan:** data polis (RDBMS/data lake) dibaca Spark, lalu disimpan ulang sebagai *Parquet* (columnar, terkompresi, cepat untuk query analitik).
    - **Pengambilan data:** dengan `createOrReplaceTempView` data dapat di-query menggunakan **Spark SQL**, mis. `SELECT AGE, AVG(OUTCOME) FROM claims GROUP BY AGE`.
    - **Integrasi sistem:** hasil prediksi dapat ditulis kembali ke basis data (PostgreSQL/MySQL via JDBC) untuk dipakai sistem underwriting & dashboard.
    """)

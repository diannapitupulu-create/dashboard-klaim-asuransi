import json, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = os.environ.get("OUT_DIR", "output")
st.set_page_config(page_title="Prediksi Klaim Asuransi Kendaraan", page_icon="🚗", layout="wide")

# ==========================================
# UPGRADE CSS (Sembunyiin Menu Streamlit & Font Modern)
# ==========================================
st.markdown("""
    <style>
    /* Import Font Google (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sembunyiin menu bawaan Streamlit (garis tiga, tulisan made with, dll) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Bikin kotak Metrik lebih estetik */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        border-left: 5px solid #FF4B4B; /* Aksen warna merah di kiri */
        transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px); /* Efek naik dikit pas dihover */
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: #1E293B !important; /* Warna text lebih elegan */
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #64748B !important;
    }

    /* Jarak antar tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi bantuan buat bikin grafik jadi transparan & bersih
def clean_plotly_layout(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=10, r=10),
        title_font=dict(size=16, color="#1E293B"),
        font=dict(family="Inter, sans-serif")
    )
    fig.update_xaxes(showgrid=False, linecolor='rgba(0,0,0,0)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)', linecolor='rgba(0,0,0,0)')
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

st.title("🚗 Logistic Regression vs Random Forest")
st.caption("Prediksi klaim asuransi kendaraan bermotor menggunakan PySpark • Dataset: Car Insurance Claim (Kaggle)")
st.write("") # Spasi

tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan Data", "🔎 Eksplorasi (EDA)", "🤖 Perbandingan Model", "🗄️ Data & Basis Data"])

# ---------------- Tab 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Data", f"{S['n_rows']:,}")
    c2.metric("Jumlah Variabel", S["n_cols"])
    claim = S["target"]["1"]
    c3.metric("Tingkat Klaim", f"{claim / S['n_rows'] * 100:.1f}%")
    c4.metric("Model Terbaik (ROC-AUC)", S["best_model"])

    st.write("---")

    l, r = st.columns([1, 1.5])
    
    tdf = pd.DataFrame({"Status": ["Tidak klaim", "Klaim"], "Jumlah": [S["target"]["0"], S["target"]["1"]]})
    fig_pie = px.pie(tdf, names="Status", values="Jumlah", hole=0.6, 
                     title="Distribusi Target (OUTCOME)",
                     color_discrete_sequence=["#94a3b8", "#ef4444"])
    fig_pie.update_traces(textposition='inside', textinfo='percent')
    fig_pie = clean_plotly_layout(fig_pie)
    fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
    l.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)

    miss = D["missing"][D["missing"]["missing"] > 0].reset_index().rename(columns={"index": "variabel"})
    fig_bar = px.bar(miss, x="variabel", y="pct", text="pct", 
                     title="Persentase Missing Value",
                     color_discrete_sequence=["#f59e0b"])
    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_bar = clean_plotly_layout(fig_bar)
    fig_bar.update_layout(yaxis_title="Persentase (%)", xaxis_title="")
    r.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    st.info(f"💡 **Info:** Duplikat baris: {S['n_dup']} • Data Train: {S['train_n']:,} • Data Test: {S['test_n']:,}")

# ---------------- Tab 2
with tab2:
    eda = D["eda"]
    var = st.selectbox("Pilih variabel kategorikal", eda["variabel"].unique(), index=0)
    sub = eda[eda["variabel"] == var]
    a, b = st.columns(2)
    
    fig_eda1 = px.bar(sub, x="kategori", y="jumlah", title=f"Jumlah Pemegang Polis per {var}",
                      color_discrete_sequence=["#3b82f6"])
    fig_eda1 = clean_plotly_layout(fig_eda1)
    a.plotly_chart(fig_eda1, use_container_width=True, config=PLOTLY_CONFIG)
    
    fig_eda2 = px.bar(sub, x="kategori", y="claim_rate", text="claim_rate",
                      title=f"Tingkat Klaim (%) per {var}", color_discrete_sequence=["#ef4444"])
    fig_eda2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_eda2 = clean_plotly_layout(fig_eda2)
    b.plotly_chart(fig_eda2, use_container_width=True, config=PLOTLY_CONFIG)

    st.write("---")
    cl = D["clean"]
    num = st.selectbox("Distribusi variabel numerik", ["CREDIT_SCORE", "ANNUAL_MILEAGE", "SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS"])
    cl2 = cl.assign(Status=cl["OUTCOME"].map({0: "Tidak klaim", 1: "Klaim"}))
    
    fig_hist = px.histogram(cl2, x=num, color="Status", barmode="overlay", opacity=0.7,
                            title=f"Distribusi {num} berdasarkan status klaim",
                            color_discrete_map={"Tidak klaim": "#94a3b8", "Klaim": "#ef4444"})
    fig_hist = clean_plotly_layout(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CONFIG)
    
    fig_corr = px.imshow(D["corr"], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Heatmap Korelasi", aspect="auto")
    fig_corr = clean_plotly_layout(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------- Tab 3
with tab3:
    st.subheader("Metrik evaluasi pada data uji")
    show = M[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns[1:]}).highlight_max(
        subset=list(show.columns[1:]), color="#bbf7d0"), use_container_width=True, hide_index=True)

    long = show.melt(id_vars="model", var_name="metrik", value_name="nilai")
    fig_metric = px.bar(long, x="metrik", y="nilai", color="model", barmode="group", text_auto=".3f",
                        title="Perbandingan Metrik", range_y=[0, 1.1],
                        color_discrete_sequence=["#3b82f6", "#10b981"])
    fig_metric.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_metric = clean_plotly_layout(fig_metric)
    st.plotly_chart(fig_metric, use_container_width=True, config=PLOTLY_CONFIG)

    st.write("---")
    a, b = st.columns(2)
    
    fig_roc = go.Figure()
    for m, g in D["roc"].groupby("model"):
        auc = M.loc[M.model == m, "roc_auc"].iloc[0]
        fig_roc.add_trace(go.Scatter(x=g["fpr"], y=g["tpr"], mode="lines", name=f"{m} (AUC={auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Acak"))
    fig_roc.update_layout(title="Kurva ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    fig_roc = clean_plotly_layout(fig_roc)
    a.plotly_chart(fig_roc, use_container_width=True, config=PLOTLY_CONFIG)

    mdl = b.radio("Pilih model untuk Confusion Matrix:", M["model"].tolist(), horizontal=True)
    row = M[M.model == mdl].iloc[0]
    cm = pd.DataFrame([[row.tn, row.fp], [row.fn, row.tp]], index=["Aktual: Tidak", "Aktual: Klaim"],
                      columns=["Prediksi: Tidak", "Prediksi: Klaim"])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=f"Confusion Matrix – {mdl}")
    fig_cm = clean_plotly_layout(fig_cm)
    b.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)

    st.write("---")
    st.subheader("Variabel Paling Berpengaruh")
    fm = st.radio("Pilih Model untuk Feature Importance:", ["Random Forest", "Logistic Regression"], horizontal=True, key="fi")
    f = D["fi"][D["fi"].model == fm].sort_values("abs", ascending=False).head(10).iloc[::-1]
    lab = "Importance" if fm == "Random Forest" else "Koefisien"
    
    fig_fi = px.bar(f, x="nilai", y="fitur", orientation="h", title=f"Top 10 Fitur – {fm}",
                    labels={"nilai": lab}, color_discrete_sequence=["#8b5cf6"])
    fig_fi = clean_plotly_layout(fig_fi)
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

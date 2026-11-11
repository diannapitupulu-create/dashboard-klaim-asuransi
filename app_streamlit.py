import json, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = os.environ.get("OUT_DIR", "output")
st.set_page_config(page_title="Prediksi Klaim Asuransi Kendaraan", page_icon="🚗", layout="wide")

# ==========================================
# CSS CUSTOM UNTUK MEMPERCANTIK UI
# ==========================================
st.markdown("""
    <style>
    /* Kecilin ukuran font metrik */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #FF4B4B;
    }
    /* Bikin kotak (card) buat metrik */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.05);
    }
    /* Rapihin judul tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Konfigurasi grafik biar toolbar (camera, zoom) ilang
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

tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan Data", "🔎 Eksplorasi (EDA)", "🤖 Perbandingan Model", "🗄️ Data & Basis Data"])

# ---------------- Tab 1
with tab1:
    # 4 Kolom untuk Metrik
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Data", f"{S['n_rows']:,}")
    c2.metric("Jumlah Variabel", S["n_cols"])
    claim = S["target"]["1"]
    c3.metric("Tingkat Klaim", f"{claim / S['n_rows'] * 100:.1f}%")
    c4.metric("Model Terbaik (ROC-AUC)", S["best_model"])
    
    st.write("---") # Garis pemisah

    # Layout: Kiri (Pie) lebih kecil, Kanan (Bar) lebih lebar
    l, r = st.columns([1, 1.5])
    
    tdf = pd.DataFrame({"Status": ["Tidak klaim", "Klaim"], "Jumlah": [S["target"]["0"], S["target"]["1"]]})
    fig_pie = px.pie(tdf, names="Status", values="Jumlah", hole=0.6, 
                     title="Distribusi Target (OUTCOME)",
                     color_discrete_sequence=["#94a3b8", "#ef4444"])
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    l.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)

    miss = D["missing"][D["missing"]["missing"] > 0].reset_index().rename(columns={"index": "variabel"})
    fig_bar = px.bar(miss, x="variabel", y="pct", text="pct", 
                     title="Persentase Missing Value",
                     color_discrete_sequence=["#f59e0b"])
    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_bar.update_layout(yaxis_title="Persentase (%)", xaxis_title="", margin=dict(t=40, b=0, l=0, r=0))
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
    a.plotly_chart(fig_eda1, use_container_width=True, config=PLOTLY_CONFIG)
    
    fig_eda2 = px.bar(sub, x="kategori", y="claim_rate", text="claim_rate",
                      title=f"Tingkat Klaim (%) per {var}", color_discrete_sequence=["#ef4444"])
    fig_eda2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    b.plotly_chart(fig_eda2, use_container_width=True, config=PLOTLY_CONFIG)

    st.write("---")
    cl = D["clean"]
    num = st.selectbox("Distribusi variabel numerik", ["CREDIT_SCORE", "ANNUAL_MILEAGE", "SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS"])
    cl2 = cl.assign(Status=cl["OUTCOME"].map({0: "Tidak klaim", 1: "Klaim"}))
    
    fig_hist = px.histogram(cl2, x=num, color="Status", barmode="overlay", opacity=0.7,
                            title=f"Distribusi {num} berdasarkan status klaim",
                            color_discrete_map={"Tidak klaim": "#94a3b8", "Klaim": "#ef4444"})
    st.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CONFIG)
    
    fig_corr = px.imshow(D["corr"], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Heatmap Korelasi", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

# ---------------- Tab 3
with tab3:
    st.subheader("Metrik evaluasi pada data uji")
    show = M[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns[1:]}).highlight_max(
        subset=list(show.columns[1:]), color="#bbf7d0"), use_container_width=True, hide_index=True)

    long = show.melt(id_vars="model", var_name="metrik", value_name="nilai")
    fig_metric = px.bar(long, x="metrik", y="nilai", color="model", barmode="group", text_auto=".3f",
                        title="Perbandingan Metrik (Accuracy, Precision, Recall, F1, ROC-AUC)", range_y=[0, 1.1],
                        color_discrete_sequence=["#3b82f6", "#10b981"])
    fig_metric.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_metric, use_container_width=True, config=PLOTLY_CONFIG)

    st.write("---")
    a, b = st.columns(2)
    
    fig_roc = go.Figure()
    for m, g in D["roc"].groupby("model"):
        auc = M.loc[M.model == m, "roc_auc"].iloc[0]
        fig_roc.add_trace(go.Scatter(x=g["fpr"], y=g["tpr"], mode="lines", name=f"{m} (AUC={auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Acak"))
    fig_roc.update_layout(title="Kurva ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          margin=dict(t=40, b=0, l=0, r=0))
    a.plotly_chart(fig_roc, use_container_width=True, config=PLOTLY_CONFIG)

    mdl = b.radio("Pilih model untuk Confusion Matrix:", M["model"].tolist(), horizontal=True)
    row = M[M.model == mdl].iloc[0]
    cm = pd.DataFrame([[row.tn, row.fp], [row.fn, row.tp]], index=["Aktual: Tidak", "Aktual: Klaim"],
                      columns=["Prediksi: Tidak", "Prediksi: Klaim"])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=f"Confusion Matrix – {mdl}")
    fig_cm.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    b.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)

    st.write("---")
    st.subheader("Variabel Paling Berpengaruh")
    fm = st.radio("Pilih Model untuk Feature Importance:", ["Random Forest", "Logistic Regression"], horizontal=True, key="fi")
    f = D["fi"][D["fi"].model == fm].sort_values("abs", ascending=False).head(10).iloc[::-1]
    lab = "Importance" if fm == "Random Forest" else "Koefisien"
    
    fig_fi = px.bar(f, x="nilai", y="fitur", orientation="h", title=f"Top 10 Fitur – {fm}",
                    labels={"nilai": lab}, color_discrete_sequence=["#8b5cf6"])
    fig_fi.update_layout(margin=dict(t=40, b=0, l=0, r=0))
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

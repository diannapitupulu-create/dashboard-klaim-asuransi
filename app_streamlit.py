import json, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = os.environ.get("OUT_DIR", "output")
st.set_page_config(page_title="Prediksi Klaim Asuransi Kendaraan", page_icon="🚗", layout="wide")


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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah data", f"{S['n_rows']:,}")
    c2.metric("Jumlah variabel", S["n_cols"])
    claim = S["target"]["1"]
    c3.metric("Tingkat klaim", f"{claim / S['n_rows'] * 100:.1f}%")
    c4.metric("Model terbaik (ROC-AUC)", S["best_model"])

    l, r = st.columns(2)
    tdf = pd.DataFrame({"Status": ["Tidak klaim", "Klaim"], "Jumlah": [S["target"]["0"], S["target"]["1"]]})
    l.plotly_chart(px.pie(tdf, names="Status", values="Jumlah", hole=0.5, title="Distribusi target (OUTCOME)",
                          color_discrete_sequence=["#94a3b8", "#ef4444"]), width="stretch")
    miss = D["missing"][D["missing"]["missing"] > 0].reset_index().rename(columns={"index": "variabel"})
    r.plotly_chart(px.bar(miss, x="variabel", y="pct", text="pct", title="Missing value (%)",
                          color_discrete_sequence=["#f59e0b"]), width="stretch")
    st.info(f"Duplikat baris: {S['n_dup']} • Train: {S['train_n']:,} • Test: {S['test_n']:,}")

# ---------------- Tab 2
with tab2:
    eda = D["eda"]
    var = st.selectbox("Pilih variabel", eda["variabel"].unique(), index=0)
    sub = eda[eda["variabel"] == var]
    a, b = st.columns(2)
    a.plotly_chart(px.bar(sub, x="kategori", y="jumlah", title=f"Jumlah pemegang polis per {var}",
                          color_discrete_sequence=["#3b82f6"]), width="stretch")
    b.plotly_chart(px.bar(sub, x="kategori", y="claim_rate", text="claim_rate",
                          title=f"Tingkat klaim (%) per {var}", color_discrete_sequence=["#ef4444"]),
                   width="stretch")

    cl = D["clean"]
    num = st.selectbox("Distribusi variabel numerik", ["CREDIT_SCORE", "ANNUAL_MILEAGE", "SPEEDING_VIOLATIONS",
                                                       "DUIS", "PAST_ACCIDENTS"])
    cl2 = cl.assign(Status=cl["OUTCOME"].map({0: "Tidak klaim", 1: "Klaim"}))
    st.plotly_chart(px.histogram(cl2, x=num, color="Status", barmode="overlay", opacity=0.65,
                                 title=f"Distribusi {num} berdasarkan status klaim",
                                 color_discrete_map={"Tidak klaim": "#94a3b8", "Klaim": "#ef4444"}),
                    width="stretch")
    st.plotly_chart(px.imshow(D["corr"], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Korelasi (numerik & biner)", aspect="auto"), width="stretch")

# ---------------- Tab 3
with tab3:
    st.subheader("Metrik evaluasi pada data uji")
    show = M[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    st.dataframe(show.style.format({c: "{:.4f}" for c in show.columns[1:]}).highlight_max(
        subset=list(show.columns[1:]), color="#bbf7d0"), width="stretch", hide_index=True)

    long = show.melt(id_vars="model", var_name="metrik", value_name="nilai")
    st.plotly_chart(px.bar(long, x="metrik", y="nilai", color="model", barmode="group", text_auto=".3f",
                           title="Perbandingan metrik", range_y=[0, 1],
                           color_discrete_sequence=["#3b82f6", "#10b981"]), width="stretch")

    a, b = st.columns(2)
    fig = go.Figure()
    for m, g in D["roc"].groupby("model"):
        auc = M.loc[M.model == m, "roc_auc"].iloc[0]
        fig.add_trace(go.Scatter(x=g["fpr"], y=g["tpr"], mode="lines", name=f"{m} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Acak"))
    fig.update_layout(title="Kurva ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    a.plotly_chart(fig, width="stretch")

    mdl = b.radio("Confusion matrix", M["model"].tolist(), horizontal=True)
    row = M[M.model == mdl].iloc[0]
    cm = pd.DataFrame([[row.tn, row.fp], [row.fn, row.tp]], index=["Aktual: Tidak", "Aktual: Klaim"],
                      columns=["Prediksi: Tidak", "Prediksi: Klaim"])
    b.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                             title=f"Confusion matrix – {mdl}"), width="stretch")

    st.subheader("Variabel paling berpengaruh")
    fm = st.radio("Model", ["Random Forest", "Logistic Regression"], horizontal=True, key="fi")
    f = D["fi"][D["fi"].model == fm].sort_values("abs", ascending=False).head(10).iloc[::-1]
    lab = "Importance" if fm == "Random Forest" else "Koefisien (data distandarisasi)"
    st.plotly_chart(px.bar(f, x="nilai", y="fitur", orientation="h", title=f"Top 10 fitur – {fm}",
                           labels={"nilai": lab}, color_discrete_sequence=["#8b5cf6"]), width="stretch")
    st.caption(f"Hyperparameter terbaik (3-fold CV) → LR: {S['lr_params']} • RF: {S['rf_params']}")

# ---------------- Tab 4
with tab4:
    st.subheader("Dataset hasil olahan")
    st.dataframe(D["clean"].head(200), width="stretch")
    st.download_button("⬇️ Unduh dataset olahan (CSV)", D["clean"].to_csv(index=False).encode(),
                       "car_insurance_clean.csv", "text/csv")
    st.markdown("""
**Keterkaitan dengan basis data**
- **Penyimpanan:** data polis (RDBMS/data lake) → dibaca Spark → disimpan ulang sebagai *Parquet* (columnar, terkompresi, cepat untuk query analitik).
- **Pengambilan data:** dengan `createOrReplaceTempView` data dapat di-query menggunakan **Spark SQL**, mis. `SELECT AGE, AVG(OUTCOME) FROM claims GROUP BY AGE`.
- **Integrasi sistem:** hasil prediksi dapat ditulis kembali ke basis data (PostgreSQL/MySQL via JDBC) untuk dipakai sistem underwriting & dashboard.
""")

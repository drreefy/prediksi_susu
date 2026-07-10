import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import shap
from sklearn.preprocessing import StandardScaler
import joblib, json, os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sistem Monitoring Kualitas Susu KUD Sarwa Mukti Cisarua",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{padding-top:1.5rem;padding-bottom:1rem}
.metric-card{background:#f8f9fa;border-radius:12px;padding:1rem 1.25rem;border:1px solid #e9ecef;text-align:center}
.metric-label{font-size:13px;color:#6c757d;margin-bottom:4px}
.metric-value{font-size:26px;font-weight:600;color:#212529}
.metric-sub{font-size:12px;color:#adb5bd;margin-top:2px}
.section-title{font-size:16px;font-weight:600;color:#212529;margin:1.2rem 0 0.4rem}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# SISTEM LOGIN
# ════════════════════════════════════════════
USERS = {
    "admin":   {"password": "admin123",   "role": "admin"},
    "petugas": {"password": "petugas123", "role": "petugas"},
}

def login_page():
    st.markdown("""
    <div style='max-width:600px;margin:4rem auto 0;text-align:center'>
        <div style='font-size:64px'>🥛</div>
        <h2 style='margin:0.5rem 0;line-height:1.3;font-size:26px;white-space:nowrap'>Sistem Monitoring Kualitas Susu<br>KUD Sarwa Mukti Cisarua</h2>
        <p style='color:#6c757d;margin-bottom:2rem'>Silakan login untuk melanjutkan</p>
    </div>""", unsafe_allow_html=True)

    col_c = st.columns([1,2,1])[1]
    with col_c:
        with st.container():
            st.markdown("<div style='background:#f8f9fa;padding:2rem;border-radius:12px;"
                        "border:1px solid #e9ecef'>", unsafe_allow_html=True)
            username = st.text_input("👤 Username", placeholder="Masukkan username")
            password = st.text_input("🔒 Password", type="password", placeholder="Masukkan password")
            if st.button("Login", use_container_width=True, type="primary"):
                user = USERS.get(username)
                if user and user["password"] == password:
                    st.session_state["logged_in"] = True
                    st.session_state["username"]  = username
                    st.session_state["role"]      = user["role"]
                    st.rerun()
                else:
                    st.error("Username atau password salah.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;margin-top:1.5rem;color:#adb5bd;font-size:13px'>
    🔑 Admin: akses penuh &nbsp;|&nbsp; Petugas: overview, prediksi, dan segmentasi
    </div>""", unsafe_allow_html=True)

# Cek login state
if not st.session_state.get("logged_in"):
    login_page()
    st.stop()

ROLE = st.session_state["role"]

# ── Kolom yang di-drop saat modeling ──
DROP_COLS = ["Tgl","Nama","NoPol","Segel","Bulan","Appearance","TDO",
             "AT","BTB","CT","Antibiotik","Bulan_Str"]
FEAT_COLS = ["Netto","Temp.","PH","TS","SNF","FAT","Density","Durasi_Menit"]
TARGET    = "TPC"

C_BLUE   = "#4A90D9"
C_ORANGE = "#e67e22"
C_PURPLE = "#9b59b6"
C_RED    = "#e74c3c"
C_GREEN  = "#2ecc71"
C_GRAY   = "#adb5bd"

# ── Load & cache ──
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df["Tgl"] = pd.to_datetime(df["Tgl"], errors="coerce")
    df["Bulan_Str"] = df["Tgl"].dt.to_period("M").astype(str)
    return df

@st.cache_resource
def load_model(df):
    MODEL_PATH = "model.pkl"
    FEAT_PATH  = "feature_names.json"

    pkl_loaded = False
    if os.path.exists(MODEL_PATH) and os.path.exists(FEAT_PATH):
        try:
            mdl = joblib.load(MODEL_PATH)
            with open(FEAT_PATH) as f:
                feat_names = json.load(f)
            source     = "💾 Model dimuat dari model.pkl"
            pkl_loaded = True
        except Exception:
            pkl_loaded = False

    if not pkl_loaded:
        feat_names = FEAT_COLS
        mdl = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        d   = df[feat_names + [TARGET]].dropna()
        mdl.fit(d[feat_names], np.log1p(d[TARGET]))
        if os.path.exists(MODEL_PATH):
            source = "⚠️ model.pkl tidak kompatibel dengan versi library saat ini — model dilatih ulang dari data"
        else:
            source = "⚙️ model.pkl tidak ditemukan — model dilatih ulang dari data"

    d      = df[feat_names + [TARGET]].dropna()
    X      = d[feat_names]
    y      = np.log1p(d[TARGET])
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = mdl.predict(X_te)
    return mdl, feat_names, X_te, y_te, y_pred, source

IMG_LEFT  = "https://cdn.pixabay.com/photo/2024/07/26/10/38/animal-8923235_640.png"
IMG_RIGHT = "https://i.pinimg.com/736x/e6/e7/e7/e6e7e7d811a544d75221b6652ab10fa3.jpg"

# Inisialisasi uploaded2 dulu sebelum dipakai di mana pun
uploaded2 = None
df_raw    = None

if ROLE == "admin":
    col_l, col_mid, col_r = st.columns([1, 1.5, 1])
    with col_l:
        st.markdown("<div style='height:100%;padding-top:3rem; padding-left: 10rem'>", unsafe_allow_html=True)
        st.image(IMG_LEFT, width=280)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_mid:
        st.markdown("""
        <div style='text-align:center;padding:2rem 0 1.5rem'>
            <div style='font-size:100px'>🥛</div>
            <h2 style='margin:0.25rem 0 0.25rem;line-height:1.3;font-size:26px;white-space:nowrap'>Sistem Monitoring Kualitas Susu<br>KUD Sarwa Mukti Cisarua</h2>
            <p style='color:#6c757d;font-size:15px;margin-bottom:1.5rem'>
                Upload file <b>data xlsx</b> untuk memulai.
            </p>
        </div>""", unsafe_allow_html=True)
        uploaded2 = st.file_uploader("Upload data", type=["xlsx"], key="center_upload")
        if uploaded2:
            df_raw = load_data(uploaded2)
    with col_r:
        st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;padding-top:5rem'>", unsafe_allow_html=True)
        st.image(IMG_RIGHT, width=230)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    # Petugas: landing page sama persis dengan admin, upload di tengah
    col_l, col_mid, col_r = st.columns([1, 1.5, 1])
    with col_l:
        st.markdown("<div style='height:100%;padding-top:3rem;padding-left:10rem'>", unsafe_allow_html=True)
        st.image(IMG_LEFT, width=280)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_mid:
        st.markdown("""
        <div style='text-align:center;padding:2rem 0 1.5rem'>
            <div style='font-size:100px'>🥛</div>
            <h2 style='margin:0.25rem 0 0.25rem;line-height:1.3;font-size:26px;white-space:nowrap'>Sistem Monitoring Kualitas Susu<br>KUD Sarwa Mukti Cisarua</h2>
            <p style='color:#6c757d;font-size:15px;margin-bottom:1.5rem'>
                Upload file <b>data_fix.xlsx</b> untuk memulai.
            </p>
        </div>""", unsafe_allow_html=True)
        uploaded2 = st.file_uploader("Upload data.xlsx", type=["xlsx"], key="center_upload")
        if uploaded2:
            df_raw = load_data(uploaded2)
    with col_r:
        st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;padding-top:5rem'>", unsafe_allow_html=True)
        st.image(IMG_RIGHT, width=230)
        st.markdown("</div>", unsafe_allow_html=True)
    

# ────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🥛 Sistem Monitoring Kualitas Susu\nKUD Sarwa Mukti Cisarua")
    st.markdown("---")

    # ── Info user + Logout (selalu tampil) ──
    st.markdown(f"👤 **{st.session_state['username']}** "
                f"({'Admin' if ROLE == 'admin' else 'Petugas Koperasi'})")
    if st.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in","username","role"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown("---")

    if uploaded2:
        if ROLE == "admin":
            df_raw = load_data(uploaded2)
        months_all = sorted(df_raw["Bulan_Str"].dropna().unique())

        st.markdown("### Filter")
        bulan_range = st.select_slider("Rentang Bulan", options=months_all,
                                       value=(months_all[0], months_all[-1]))
        if ROLE == "admin":
            kud_opts   = ["Semua"] + sorted(df_raw["Nama"].dropna().unique().tolist())
            nopol_opts = ["Semua"] + sorted(df_raw["NoPol"].dropna().unique().tolist())
            sel_kud   = st.selectbox("KUD", kud_opts)
            sel_nopol = st.selectbox("NoPol", nopol_opts)
        else:
            sel_kud   = "Semua"
            sel_nopol = "Semua"

        st.markdown("---")

        # Halaman sesuai role
        if ROLE == "admin":
            halaman_opts = [
                "📊 Overview",
                "📈 Statistika Deskriptif",
                "🔍 Eksplorasi Data",
                "🤖 Model Performance",
                "🧪 Simulasi Prediksi",
                "🔬 Segmentasi TPK",
            ]
        else:
            halaman_opts = [
                "📊 Overview",
                "🧪 Simulasi Prediksi",
                "🔬 Segmentasi TPK",
            ]

        page = st.radio("Halaman", halaman_opts)
    else:
        page = None

# ── No data ──
if not uploaded2:
     
    st.stop()

# ── Apply filter ──
df = df_raw[(df_raw["Bulan_Str"] >= bulan_range[0]) & (df_raw["Bulan_Str"] <= bulan_range[1])].copy()
if sel_kud   != "Semua": df = df[df["Nama"]  == sel_kud]
if sel_nopol != "Semua": df = df[df["NoPol"] == sel_nopol]

months_f = sorted(df["Bulan_Str"].dropna().unique())
monthly = (
    df.groupby("Bulan_Str")[TARGET]
    .agg(Jumlah="count", Mean="mean", Median="median", Std="std",
         Min="min", Max="max",
         Q1=lambda x: x.quantile(0.25), Q3=lambda x: x.quantile(0.75))
    .reindex(months_f).round(4)
)

def metric_card(label, value, sub="", color="#212529", border=None):
    border_style = f"border:2px solid {border};" if border else ""
    return f"""<div class='metric-card' style='{border_style}'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value' style='color:{color}'>{value}</div>
        <div class='metric-sub'>{sub}</div>
    </div>"""

# ════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("## 📊 Overview")
    st.caption(f"Periode **{bulan_range[0]}** s/d **{bulan_range[1]}** · {len(df):,} data")

    c1,c2,c3,c4,c5 = st.columns(5)
    bulan_best  = monthly["Mean"].idxmin() if not monthly.empty else "-"
    bulan_worst = monthly["Mean"].idxmax() if not monthly.empty else "-"
    for col, lbl, val, sub in zip(
        [c1,c2,c3,c4,c5],
        ["Total Data","Rata-Rata TPC","Median TPC","Bulan Terbaik ↓","Bulan Terburuk ↑"],
        [f"{len(df):,}", f"{df[TARGET].mean():.3f}", f"{df[TARGET].median():.3f}", bulan_best, bulan_worst],
        ["Baris","CFU/mL","CFU/mL","Mean TPC Terendah","Mean TPC Tertinggi"],
    ):
        col.markdown(metric_card(lbl, val, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([2,1])

    with cl:
        st.markdown("<div class='section-title'>Tren TPC per Bulan</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Max"], mode="lines",
            line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Min"], mode="lines",
            fill="tonexty", fillcolor="rgba(74,144,217,0.08)", line=dict(width=0), name="Min–Max"))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Mean"]+monthly["Std"],
            mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Mean"]-monthly["Std"],
            mode="lines", fill="tonexty", fillcolor="rgba(74,144,217,0.18)",
            line=dict(width=0), name="±1 Std"))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Median"],
            mode="lines+markers", line=dict(color=C_ORANGE, width=2, dash="dash"),
            marker=dict(size=5), name="Median"))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Mean"],
            mode="lines+markers+text", line=dict(color=C_BLUE, width=2.5),
            marker=dict(size=7),
            text=[f"{v:.2f}" for v in monthly["Mean"]], textposition="top center",
            textfont=dict(size=9), name="Mean"))
        fig.update_layout(height=320, margin=dict(t=10,b=10,l=0,r=0),
            xaxis_title="Bulan", yaxis_title="TPC (CFU/mL)",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown("<div class='section-title'>Distribusi TPC</div>", unsafe_allow_html=True)
        fig_h = px.histogram(df, x=TARGET, nbins=35, color_discrete_sequence=[C_BLUE])
        fig_h.update_layout(height=320, margin=dict(t=10,b=10,l=0,r=0),
            bargap=0.05, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
            showlegend=False, xaxis_title="TPC (CFU/mL)", yaxis_title="Frekuensi")
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("<div class='section-title'>Jumlah Data per Bulan</div>", unsafe_allow_html=True)
    fig_b = px.bar(monthly.reset_index(), x="Bulan_Str", y="Jumlah",
        labels={"Bulan_Str":"Bulan","Jumlah":"Jumlah Data"}, text="Jumlah",
        color_discrete_sequence=[C_BLUE])
    fig_b.update_traces(textposition="outside")
    fig_b.update_layout(height=270, margin=dict(t=10,b=10,l=0,r=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"))
    st.plotly_chart(fig_b, use_container_width=True)

# ════════════════════════════════════════════
# PAGE 2 — STATISTIKA DESKRIPTIF
# ════════════════════════════════════════════
elif page == "📈 Statistika Deskriptif":
    st.markdown("## 📈 Statistika Deskriptif")

    tc1,tc2,tc3,tc4,tc5 = st.columns(5)
    show_mean   = tc1.checkbox("Mean",    value=True)
    show_median = tc2.checkbox("Median",  value=True)
    show_std    = tc3.checkbox("±1 Std",  value=True)
    show_iqr    = tc4.checkbox("IQR",     value=True)
    show_minmax = tc5.checkbox("Min–Max", value=False)

    fig = go.Figure()
    if show_minmax:
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Max"], mode="lines",
            line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Min"], mode="lines",
            fill="tonexty", fillcolor="rgba(173,181,189,0.15)", line=dict(width=0), name="Min–Max"))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Min"], mode="lines+markers",
            line=dict(color=C_GRAY, dash="dot", width=1), marker=dict(size=3, symbol="triangle-down"), name="Min"))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Max"], mode="lines+markers",
            line=dict(color=C_GRAY, dash="dot", width=1), marker=dict(size=3, symbol="triangle-up"), name="Max"))
    if show_std:
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Mean"]+monthly["Std"],
            mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Mean"]-monthly["Std"],
            mode="lines", fill="tonexty", fillcolor="rgba(74,144,217,0.18)",
            line=dict(width=0), name="±1 Std"))
    if show_iqr:
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Q3"], mode="lines",
            line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Q1"], mode="lines",
            fill="tonexty", fillcolor="rgba(155,89,182,0.20)", line=dict(width=0), name="IQR (Q1–Q3)"))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Q1"], mode="lines+markers",
            line=dict(color=C_PURPLE, dash="dash", width=1.2), marker=dict(size=4, symbol="triangle-down"), name="Q1"))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Q3"], mode="lines+markers",
            line=dict(color=C_PURPLE, dash="dash", width=1.2), marker=dict(size=4, symbol="triangle-up"), name="Q3"))
    if show_median:
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Median"], mode="lines+markers",
            line=dict(color=C_ORANGE, width=2, dash="dash"), marker=dict(size=5), name="Median"))
    if show_mean:
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Mean"],
            mode="lines+markers+text", line=dict(color=C_BLUE, width=2.5),
            marker=dict(size=7),
            text=[f"{v:.2f}" for v in monthly["Mean"]], textposition="top center",
            textfont=dict(size=9), name="Mean"))

    fig.update_layout(height=420, margin=dict(t=20,b=10,l=0,r=0),
        xaxis_title="Bulan", yaxis_title="TPC (CFU/mL)",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Tabel Ringkasan</div>", unsafe_allow_html=True)
    st.dataframe(monthly.style.format("{:.4f}").background_gradient(subset=["Mean"], cmap="Blues"),
                 use_container_width=True)

# ════════════════════════════════════════════
# PAGE 3 — EKSPLORASI DATA
# ════════════════════════════════════════════
elif page == "🔍 Eksplorasi Data":
    st.markdown("## 🔍 Eksplorasi Data")

    # ── Filter Tahun ──
    tahun_opts_ex = ["Semua"] + sorted(df["Tgl"].dt.year.dropna().unique().astype(int).tolist())
    sel_tahun_ex  = st.selectbox("Filter Tahun", tahun_opts_ex, index=0, key="eksplorasi_tahun")
    df_ex = df[df["Tgl"].dt.year == sel_tahun_ex].copy() if sel_tahun_ex != "Semua" else df.copy()
    st.caption(f"{len(df_ex):,} data ditampilkan" + (f" · Tahun {sel_tahun_ex}" if sel_tahun_ex != "Semua" else " · Semua tahun"))

    num_cols = FEAT_COLS + [TARGET]
    cl, cr = st.columns(2)
    x_ax = cl.selectbox("Sumbu X", num_cols, index=0)
    y_ax = cr.selectbox("Sumbu Y", num_cols, index=num_cols.index(TARGET))

    df_ex["Tahun"] = df_ex["Tgl"].dt.year.astype(str)
    cb_opts = ["Tahun", "Bulan_Str", "Nama", "NoPol"]
    color_by = st.selectbox("Warna berdasarkan", cb_opts)

    BRIGHT_COLORS = ["#e6000a","#007bff","#00b300","#ff8c00","#9400d3","#00bcd4","#ff1493","#8B4513"]
    # Scatter tanpa trendline bawaan plotly (butuh statsmodels)
    fig_s = px.scatter(df_ex, x=x_ax, y=y_ax, color=color_by,
        hover_data=["Bulan_Str","Nama","NoPol"],
        color_discrete_sequence=BRIGHT_COLORS)
    fig_s.update_traces(marker=dict(size=8, opacity=1,
                                    line=dict(width=0.3, color="white")))
    # Tambah garis regresi manual pakai numpy
    _pair_tr = df_ex[[x_ax, y_ax]].dropna()
    if len(_pair_tr) >= 2:
        _x = _pair_tr[x_ax].values
        _y = _pair_tr[y_ax].values
        _m, _b = np.polyfit(_x, _y, 1)
        _xr = np.linspace(_x.min(), _x.max(), 100)
        _yr = _m * _xr + _b
        fig_s.add_trace(go.Scatter(x=_xr, y=_yr, mode="lines",
            line=dict(color="#000000", width=2.5, dash="dash"),
            name="Trendline", showlegend=True))
    fig_s.update_layout(height=460, margin=dict(t=40,b=10,l=0,r=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False,
                   title=dict(text=x_ax, font=dict(size=17, color="#222")),
                   tickfont=dict(size=15, color="#222")),
        yaxis=dict(gridcolor="#e0e0e0",
                   title=dict(text=y_ax, font=dict(size=17, color="#222")),
                   tickfont=dict(size=15, color="#222")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=15, color="#222"),
                    title=dict(text=color_by, font=dict(size=15))))
    st.plotly_chart(fig_s, use_container_width=True)

    # ── Analisis Korelasi ──
    pair = df_ex[[x_ax, y_ax]].dropna()
    if len(pair) >= 3 and x_ax != y_ax:
        r = pair[x_ax].corr(pair[y_ax])
        r2 = r ** 2

        abs_r = abs(r)
        if abs_r < 0.1:
            kekuatan, kekuatan_kelas = "sangat lemah / tidak ada", "#6c757d"
        elif abs_r < 0.3:
            kekuatan, kekuatan_kelas = "lemah", "#f39c12"
        elif abs_r < 0.5:
            kekuatan, kekuatan_kelas = "sedang", "#e67e22"
        elif abs_r < 0.7:
            kekuatan, kekuatan_kelas = "kuat", "#27ae60"
        else:
            kekuatan, kekuatan_kelas = "sangat kuat", "#16a085"

        arah = "positif (searah)" if r > 0 else ("negatif (berlawanan)" if r < 0 else "tidak ada arah")
        arah_penjelasan = (
            f"semakin tinggi {x_ax}, cenderung semakin tinggi pula {y_ax}" if r > 0.05 else
            f"semakin tinggi {x_ax}, cenderung semakin rendah {y_ax}" if r < -0.05 else
            f"tidak ada kecenderungan arah yang jelas antara {x_ax} dan {y_ax}"
        )

        n = len(pair)
        if n > 2 and abs_r < 1:
            t_stat = r * np.sqrt((n - 2) / (1 - r**2))
            from scipy import stats as _stats
            p_value = 2 * (1 - _stats.t.cdf(abs(t_stat), df=n-2))
        else:
            p_value = None

        st.markdown("---")
        st.markdown("<div class='section-title'>📈 Analisis Korelasi</div>", unsafe_allow_html=True)

        cA, cB, cC = st.columns(3)
        r_color = "#e74c3c" if r < 0 else "#27ae60"
        cA.markdown(metric_card("Koefisien Korelasi (r)", f"{r:.4f}", f"R² = {r2:.4f}",
                                 color=r_color, border=r_color), unsafe_allow_html=True)
        cB.markdown(metric_card("Kekuatan Hubungan", kekuatan.title(), "",
                                 color=kekuatan_kelas, border=kekuatan_kelas), unsafe_allow_html=True)
        sig_text  = "Signifikan (p < 0.05)" if (p_value is not None and p_value < 0.05) else "Tidak signifikan"
        sig_color = "#27ae60" if (p_value is not None and p_value < 0.05) else "#e74c3c"
        cC.markdown(metric_card("Signifikansi", sig_text,
                                 f"p-value = {p_value:.4f}" if p_value is not None else "n/a",
                                 color=sig_color, border=sig_color), unsafe_allow_html=True)

        p_str = f"{p_value:.4f}" if p_value is not None else "n/a"
        if p_value is not None and p_value < 0.05:
            sig_kalimat = (f"Secara statistik, hubungan ini <b>terbukti nyata</b> (p-value {p_str} &lt; 0.05) "
                           f"dan dapat dijadikan acuan dalam pengambilan keputusan.")
        else:
            sig_kalimat = (f"Secara statistik, hubungan ini <b>belum terbukti nyata</b> "
                           f"(p-value {p_str} &ge; 0.05), artinya perlu kehati-hatian dalam menarik kesimpulan "
                           f"dari pola yang terlihat.")

        # Kalimat garis regresi berdasarkan arah
        if r > 0.05:
            regresi_kalimat = (f"Garis regresi (hitam putus-putus) menunjukkan <b>tren naik</b> — "
                               f"seiring meningkatnya nilai {x_ax}, nilai {y_ax} cenderung ikut meningkat.")
        elif r < -0.05:
            regresi_kalimat = (f"Garis regresi (hitam putus-putus) menunjukkan <b>tren turun</b> — "
                               f"seiring meningkatnya nilai {x_ax}, nilai {y_ax} cenderung menurun.")
        else:
            regresi_kalimat = (f"Garis regresi (hitam putus-putus) hampir <b>mendatar</b>, "
                               f"menunjukkan tidak ada kecenderungan naik maupun turun antara {x_ax} dan {y_ax}.")

        st.markdown(
            f"<div style='background:#f8f9fa;border-left:4px solid {kekuatan_kelas};"
            f"padding:18px 22px;border-radius:6px;margin-top:14px;font-size:19px;line-height:1.7;'>"
            f"<b>Interpretasi:</b> Hubungan antara <b>{x_ax}</b> dan <b>{y_ax}</b> memiliki nilai korelasi "
            f"<b>r = {r:.4f}</b> (R² = {r2:.4f}), tergolong "
            f"<b>{kekuatan}</b> dan bersifat <b>{arah}</b>. "
            f"{regresi_kalimat} "
            f"{sig_kalimat}"
            f"</div>",
            unsafe_allow_html=True)

        # ── Line Chart Tren Variabel X per Bulan ──
        st.markdown("---")
        st.markdown(f"<div class='section-title'>📅 Tren {x_ax} per Bulan</div>",
                    unsafe_allow_html=True)
        st.caption(f"Rata-rata {x_ax} per bulan berdasarkan data yang ditampilkan.")

        tren_df = (
            df_ex[["Bulan_Str", x_ax]].dropna()
            .groupby("Bulan_Str")[x_ax].mean()
            .reset_index()
            .rename(columns={x_ax: f"Rata-rata {x_ax}"})
        )
        # Urutkan bulan secara kronologis
        tren_df = tren_df.sort_values("Bulan_Str").reset_index(drop=True)

        fig_line = px.line(tren_df, x="Bulan_Str", y=f"Rata-rata {x_ax}",
            markers=True,
            labels={"Bulan_Str": "Bulan", f"Rata-rata {x_ax}": x_ax})
        fig_line.update_traces(
            line=dict(color=C_BLUE, width=2.5),
            marker=dict(size=8, color=C_BLUE, line=dict(width=1.5, color="white")))
        fig_line.update_layout(
            height=380, margin=dict(t=20,b=20,l=0,r=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False,
                       title=dict(text="Bulan", font=dict(size=17, color="#222")),
                       tickfont=dict(size=14, color="#222"), tickangle=-30),
            yaxis=dict(gridcolor="#e0e0e0",
                       title=dict(text=f"Rata-rata {x_ax}", font=dict(size=17, color="#222")),
                       tickfont=dict(size=14, color="#222")))
        st.plotly_chart(fig_line, use_container_width=True)

    elif x_ax == y_ax:
        st.info("Pilih dua variabel yang berbeda untuk melihat analisis korelasi.")

    with st.expander("📋 Lihat Data Mentah"):
        st.dataframe(df_ex[["Tgl","Nama","NoPol","Bulan_Str"] + FEAT_COLS + [TARGET]]
                     .reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance")
    st.caption("Model dilatih menggunakan seluruh data menggunakan Random Forest (GridSearchCV).")

    mdl, feat_names, X_te, y_te, y_pred, src = load_model(df_raw)
    st.caption(src)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    y_act_orig  = np.expm1(y_te.values)
    y_pred_orig = np.expm1(y_pred)

    c1,c2,c3 = st.columns(3)
    for col, lbl, val, clr in zip([c1,c2,c3],
        ["MAE (log-space)","RMSE (log-space)","R²"],
        [f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}"],
        [C_BLUE, C_ORANGE, C_GREEN]):
        col.markdown(metric_card(lbl, val, color=clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown("<div class='section-title'>Actual vs Predicted TPC</div>", unsafe_allow_html=True)
        lmin = min(y_act_orig.min(), y_pred_orig.min()) * 0.9
        lmax = max(y_act_orig.max(), y_pred_orig.max()) * 1.1
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(x=[lmin,lmax], y=[lmin,lmax],
            mode="lines", line=dict(color=C_RED, dash="dash", width=1.5), name="Perfect"))
        fig_avp.add_trace(go.Scatter(x=y_act_orig, y=y_pred_orig, mode="markers",
            marker=dict(color=C_BLUE, size=6, opacity=0.55), name="Data test",
            hovertemplate="Aktual: %{x:.3f}<br>Prediksi: %{y:.3f}<extra></extra>"))
        fig_avp.update_layout(height=360, margin=dict(t=10,b=10,l=0,r=0),
            xaxis_title="TPC Aktual", yaxis_title="TPC Prediksi",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False, range=[lmin,lmax]),
            yaxis=dict(gridcolor="#f0f0f0", range=[lmin,lmax]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_avp, use_container_width=True)

    with cr:
        st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
        fi = pd.DataFrame({"Fitur": feat_names, "Importance": mdl.feature_importances_})
        fi = fi.sort_values("Importance", ascending=True)
        med_i = fi["Importance"].median()
        fig_fi = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Fitur"], orientation="h",
            marker_color=[C_BLUE if v > med_i else C_GRAY for v in fi["Importance"]],
            text=[f"{v:.3f}" for v in fi["Importance"]], textposition="outside"))
        fig_fi.update_layout(height=360, margin=dict(t=10,b=10,l=0,r=0),
            xaxis_title="Importance Score",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(showgrid=False))
        st.plotly_chart(fig_fi, use_container_width=True)



# ════════════════════════════════════════════
# PAGE 5 — SIMULASI PREDIKSI
# ════════════════════════════════════════════
elif page == "🧪 Simulasi Prediksi":
    st.markdown("## 🧪 Simulasi Prediksi TPC")
    st.caption("Input nilai fitur secara manual → model memprediksi TPC.")

    mdl, feat_names, X_te, y_te, y_pred, src = load_model(df_raw)
    st.caption(src)
    ref = df_raw[feat_names].dropna()

    # ── Batas normal tiap fitur sesuai Tabel 4.3 Standar Kualitas Susu Sapi ──
    FEAT_BOUNDS = {
        "Netto": {
            "min": None, "max": None, "unit": "kg", "label": "Netto",
        },
        "Temp.": {
            "min": 1, "max": 4, "unit": "°C", "label": "Suhu",
            "low_msg":  "Suhu terlalu rendah — di bawah batas standar (1–4 °C), kondisi penyimpanan perlu diperhatikan",
            "high_msg": "Suhu terlalu tinggi — melebihi batas standar (1–4 °C), susu rentan kontaminasi dan pertumbuhan bakteri",
        },
        "PH": {
            "min": 6.5, "max": 6.6, "unit": "", "label": "Derajat Keasaman (pH)",
            "low_msg":  "pH sangat asam — di bawah batas standar (6,5–6,6), susu kemungkinan sudah basi atau terkontaminasi",
            "high_msg": "pH sangat basa — di atas batas standar (6,5–6,6), susu tidak normal",
        },
        "TS": {
            "min": 11.9, "max": None, "unit": "%", "label": "Total Solid (TS)",
            "low_msg":  "Total Solid di bawah standar (min 11,9%) — susu encer, kemungkinan diencerkan air",
            "high_msg": None,
        },
        "SNF": {
            "min": 8.5, "max": 8.8, "unit": "%", "label": "Solid Non Fat (SNF)",
            "low_msg":  "SNF di bawah standar (8,5–8,8%) — kandungan padatan non-lemak kurang dari standar",
            "high_msg": "SNF di atas standar (8,5–8,8%) — kandungan padatan non-lemak melebihi batas wajar",
        },
        "FAT": {
            "min": 3.0, "max": None, "unit": "%", "label": "Kadar Lemak (FAT)",
            "low_msg":  "Kadar lemak di bawah standar (min 3,0%) — kandungan lemak kurang dari standar SNI",
            "high_msg": None,
        },
        "Density": {
            "min": None, "max": None, "unit": "", "label": "Density",
        },
        "Durasi_Menit": {
            "min": None, "max": 240, "unit": "menit", "label": "Durasi Penanganan",
            "high_msg": "Durasi terlalu lama — melebihi 240 menit (4 jam), susu berisiko menurun kualitasnya",
        },
    }

    st.markdown("<div class='section-title'>Input Fitur</div>", unsafe_allow_html=True)
    cols3 = st.columns(4)
    inputs = {}
    feat_meta = {
        "Netto":       ("Netto (kg)", ""),
        "Temp.":       ("Temperatur (°C)", ""),
        "PH":          ("pH", ""),
        "TS":          ("Total Solid (%)", ""),
        "SNF":         ("SNF (%)", ""),
        "FAT":         ("Fat (%)", ""),
        "Density":     ("Density", ""),
        "Durasi_Menit":("Durasi (menit)", ""),
    }
    # Default khusus Durasi: 210 menit (tengah rentang 3–4 jam)
    FEAT_DEFAULT = {"Durasi_Menit": 210.0}

    for i, feat in enumerate(feat_names):
        col  = cols3[i % 4]
        fmin  = float(ref[feat].min())
        fmax  = float(ref[feat].max())
        fmean = float(ref[feat].mean())
        step  = round((fmax - fmin) / 100, 4) if fmax != fmin else 0.01
        lbl   = feat_meta.get(feat, (feat,""))[0]
        default_val = FEAT_DEFAULT.get(feat, round(fmean, 4))
        inputs[feat] = col.number_input(lbl, value=default_val,
                                        step=step, format="%.2f")

        # Tandai merah jika di luar batas SNI
        bounds = FEAT_BOUNDS.get(feat, {})
        lo, hi = bounds.get("min"), bounds.get("max")
        val    = inputs[feat]
        out_low  = lo is not None and val < lo
        out_high = hi is not None and val > hi
        if out_low or out_high:
            col.markdown(
                f"<div style='color:#c0392b;font-size:11px;margin-top:-10px;"
                f"padding:3px 6px;background:#fde8e8;border-radius:4px;'>"
                f"⚠️ Di luar batas normal ({lo}–{hi} {bounds.get('unit','')})</div>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Prediksi Sekarang", use_container_width=True, type="primary"):
        inp_df   = pd.DataFrame([inputs])
        pred_log = mdl.predict(inp_df)[0]
        pred_tpc = np.expm1(pred_log)

        # SNI 3141.1:2011 — TPC ≤ 1×10⁶ CFU/mL
        THRESHOLD = 1  # Satuan data TPC = jutaan CFU/mL, jadi batas SNI = 1 (= 1.000.000 CFU/mL)
        tpc_safe  = pred_tpc <= THRESHOLD

        # ── Cek peringatan parameter SNI dulu (sebelum tampil status) ──
        peringatan = []
        for feat, val in inputs.items():
            bounds   = FEAT_BOUNDS.get(feat, {})
            lo, hi   = bounds.get("min"), bounds.get("max")
            low_msg  = bounds.get("low_msg")
            high_msg = bounds.get("high_msg")
            if lo is not None and val < lo and low_msg:
                peringatan.append(("low", feat, bounds.get("label", feat), val, lo, hi, bounds.get("unit",""), low_msg))
            elif hi is not None and val > hi and high_msg:
                peringatan.append(("high", feat, bounds.get("label", feat), val, lo, hi, bounds.get("unit",""), high_msg))

        ada_peringatan = len(peringatan) > 0

        # Status final: AMAN hanya jika TPC aman DAN semua parameter dalam batas SNI
        # PERLU EVALUASI: TPC aman tapi ada parameter di luar SNI
        # TIDAK AMAN: TPC melebihi threshold
        if not tpc_safe:
            status_label = "⚠️ TIDAK AMAN"
            status_color = C_RED
            status_note  = "TPC melebihi batas SNI 1.000.000 CFU/mL"
        elif ada_peringatan:
            status_label = "⚠️ PERLU EVALUASI"
            status_color = C_ORANGE
            status_note  = f"TPC aman, tapi {len(peringatan)} parameter di luar batas SNI"
        else:
            status_label = "✅ AMAN"
            status_color = C_GREEN
            status_note  = "TPC dan semua parameter memenuhi standar SNI"

        c1, c2 = st.columns(2)
        # ── Card Prediksi TPC — font value digedein ──
        c1.markdown(
            f"<div class='metric-card' style='border:2px solid {status_color}'>"
            f"<div class='metric-label'>Prediksi TPC</div>"
            f"<div style='font-size:42px;font-weight:800;color:{status_color};line-height:1.1'>"
            f"{pred_tpc:,.2f}</div>"
            f"<div class='metric-sub'> Juta CFU/mL</div>"
            f"</div>",
            unsafe_allow_html=True)
        c2.markdown(metric_card(
            "Status (SNI 3141.1:2011)",
            status_label, status_note,
            color=status_color, border=status_color), unsafe_allow_html=True)

        # ── SHAP: kontribusi tiap fitur spesifik untuk input ini ──
        try:
            explainer   = shap.TreeExplainer(mdl)
            shap_vals   = explainer.shap_values(inp_df)[0]
            shap_dict   = dict(zip(feat_names, shap_vals))
            shap_sorted = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

            total_abs = sum(abs(sv) for _, sv in shap_sorted) or 1

            # Buat set fitur yang di luar batas SNI
            fitur_diluar_sni = set()
            for feat, val in inputs.items():
                bounds = FEAT_BOUNDS.get(feat, {})
                lo, hi = bounds.get("min"), bounds.get("max")
                if (lo is not None and val < lo) or (hi is not None and val > hi):
                    fitur_diluar_sni.add(feat)

            if tpc_safe:
                # AMAN: tampilkan hanya fitur yang di luar SNI (merah), sisanya hijau
                shap_items = []
                for f, sv in shap_sorted:
                    diluar = f in fitur_diluar_sni
                    shap_items.append({
                        "label": feat_meta.get(f,(f,""))[0],
                        "val":   inputs[f],
                        "bg":    "#fde8e8" if diluar else "#f0fff4",
                        "bc":    "#e74c3c" if diluar else "#27ae60",
                        "sub":   "⚠️ Di luar SNI" if diluar else "✅ Normal",
                        "sub_c": "#e74c3c" if diluar else "#27ae60",
                    })
                hdr_bg, hdr_br, hdr_tx = "#f0fff4", "#27ae60", "#155724"
                hdr_judul = "✅ TPC dalam kondisi aman"
                hdr_sub   = "Fitur merah perlu diperhatikan meskipun TPC masih di bawah batas SNI"
            else:
                # TIDAK AMAN: fitur yang mendorong TPC naik ≥ 2%, merah kalau di luar SNI
                shap_items = []
                for f, sv in shap_sorted:
                    pct = abs(sv) / total_abs * 100
                    if sv > 0 and pct >= 2:
                        diluar = f in fitur_diluar_sni
                        shap_items.append({
                            "label": feat_meta.get(f,(f,""))[0],
                            "val":   inputs[f],
                            "pct":   pct,
                            "bg":    "#fde8e8" if diluar else "#fff0f0",
                            "bc":    "#c0392b" if diluar else "#e74c3c",
                            "sub":   f"🔴 {pct:.1f}% · Di luar SNI" if diluar else f"🔴 {pct:.1f}%",
                            "sub_c": "#c0392b" if diluar else "#e74c3c",
                        })
                hdr_bg, hdr_br, hdr_tx = "#fde8e8", "#c0392b", "#721c24"
                hdr_judul = "🔍 Fitur signifikan yang mendorong TPC naik"
                hdr_sub   = ""

            if shap_items:
                st.markdown(
                    f"<div style='background:{hdr_bg};border-left:4px solid {hdr_br};"
                    f"padding:10px 14px;border-radius:6px;margin:12px 0 6px 0;'>"
                    f"<b>{hdr_judul}</b><br>"
                    f"<span style='font-size:12px;color:{hdr_tx}'>{hdr_sub}</span></div>",
                    unsafe_allow_html=True)

                n_cols = 4
                rows   = [shap_items[i:i+n_cols] for i in range(0, len(shap_items), n_cols)]
                for row in rows:
                    cols_shap = st.columns(len(row))
                    for ci, item in enumerate(row):
                        _bg   = item["bg"]
                        _bc   = item["bc"]
                        _lbl  = item["label"]
                        _val  = item["val"]
                        _sub  = item["sub"]
                        _subc = item["sub_c"]
                        cols_shap[ci].markdown(
                            f"<div style='background:{_bg};border:1.5px solid {_bc};"
                            f"border-radius:10px;padding:10px 12px;text-align:center;'>"
                            f"<div style='font-size:12px;color:#555;margin-bottom:2px'>{_lbl}</div>"
                            f"<div style='font-size:20px;font-weight:700;color:#222'>{_val:.2f}</div>"
                            f"<div style='font-size:13px;font-weight:600;color:{_subc};margin-top:4px'>"
                            f"{_sub}</div>"
                            f"</div>",
                            unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"SHAP tidak tersedia: {e}")



        if peringatan:
            st.markdown("#### ⚠️ Peringatan Parameter")
            for direction, feat, lbl, val, lo, hi, unit, msg in peringatan:
                icon      = "🔵" if direction == "low" else "🔴"
                range_str = f"{lo}–{hi} {unit}".strip() if lo and hi else (f"min {lo} {unit}" if lo else f"maks {hi} {unit}")
                st.markdown(
                    f"<div style='background:#fff3cd;border-left:4px solid #f39c12;"
                    f"padding:10px 14px;border-radius:6px;margin-bottom:8px;'>"
                    f"{icon} <b>{lbl}</b>: nilai saat ini <b>{val:.2f} {unit}</b> "
                    f"(batas normal: {range_str})<br>"
                    f"<span style='color:#856404'>{msg}</span></div>",
                    unsafe_allow_html=True)

        if not tpc_safe:
            st.error(f"TPC diprediksi **{pred_tpc:,.2f} Juta CFU/mL** — melebihi batas SNI (1 juta CFU/mL). "
                     "Susu tidak layak dan perlu penanganan lebih lanjut.")
        elif ada_peringatan:
            st.warning(f"TPC diprediksi **{pred_tpc:,.2f} CFU/mL** — memenuhi standar SNI, "
                       f"namun terdapat **{len(peringatan)} parameter** di luar batas SNI yang perlu dievaluasi.")
        else:
            st.success(f"TPC diprediksi **{pred_tpc:,.2f} CFU/mL** — memenuhi standar SNI "
                       "dan seluruh parameter berada dalam batas normal.")


# ════════════════════════════════════════════
# PAGE 6 — SEGMENTASI TPK
# ════════════════════════════════════════════
elif page == "🔬 Segmentasi TPK":
    st.markdown("## 🔬 Segmentasi TPK")
    st.caption("Unggah data CSV untuk melakukan segmentasi TPK berdasarkan karakteristik fisik susu (KA, FAT, SNF, TS).")

    # ── Upload CSV ──
    uploaded_csv = st.file_uploader("Upload CSV Data TPK", type=["csv"], key="seg_upload")

    if uploaded_csv:
        df_clust_raw = pd.read_csv(uploaded_csv, low_memory=False)

        # Bersihkan koma → titik untuk kolom numerik
        for _col in ["KA", "FAT", "SNF", "TS"]:
            if _col in df_clust_raw.columns:
                if df_clust_raw[_col].dtype == object:
                    df_clust_raw[_col] = df_clust_raw[_col].astype(str).str.replace(",", ".", regex=False)
                df_clust_raw[_col] = pd.to_numeric(df_clust_raw[_col], errors="coerce")

        # Buang SAMPEL INDIVIDU
        if "NAMA_KELOMPOK" in df_clust_raw.columns:
            df_clust_raw = df_clust_raw[
                df_clust_raw["NAMA_KELOMPOK"].astype(str).str.upper() != "SAMPEL INDIVIDU"
            ].copy()

        # ── Filter Tahun ──
        CLUST_FEATS  = ["KA", "FAT", "SNF", "TS"]
        CLUST_LABELS = {0: "Kualitas Rendah 🔴", 1: "Kualitas Sedang 🟡", 2: "Kualitas Tinggi 🟢"}
        CLUST_COLORS = {
            "Kualitas Rendah 🔴": C_RED,
            "Kualitas Sedang 🟡": C_ORANGE,
            "Kualitas Tinggi 🟢": C_GREEN,
        }
        BADGE_STYLE = {
            "Kualitas Rendah 🔴": "background:#f8d7da;color:#721c24",
            "Kualitas Sedang 🟡": "background:#fff3cd;color:#856404",
            "Kualitas Tinggi 🟢": "background:#d4edda;color:#155724",
        }
        HEADER_BG = {
            "Kualitas Rendah 🔴": "#f8d7da",
            "Kualitas Sedang 🟡": "#fff3cd",
            "Kualitas Tinggi 🟢": "#d4edda",
        }
        HEADER_TX = {
            "Kualitas Rendah 🔴": "#721c24",
            "Kualitas Sedang 🟡": "#856404",
            "Kualitas Tinggi 🟢": "#155724",
        }

        tahun_ada = []
        if "TAHUN" in df_clust_raw.columns:
            tahun_ada = sorted(df_clust_raw["TAHUN"].dropna().unique().astype(int).tolist())

        if tahun_ada:
            opsi = ["Semua Tahun"] + [str(t) for t in tahun_ada]
            sel  = st.selectbox("📅 Filter Tahun", opsi, index=0, key="seg_tahun")
            df_filtered = (
                df_clust_raw[df_clust_raw["TAHUN"] == int(sel)].copy()
                if sel != "Semua Tahun" else df_clust_raw.copy()
            )
        else:
            df_filtered = df_clust_raw.copy()

        # ── K-Means ──
        @st.cache_data
        def jalankan_kmeans_custom(df_in):
            df_proc = df_in[CLUST_FEATS + ["NAMA_KELOMPOK"]].dropna().copy()
            # Filter outlier: pakai batas wajar nilai susu
            df_proc = df_proc[
                (df_proc["TS"]  >= 5)  & (df_proc["TS"]  <= 20) &
                (df_proc["KA"]  >= 0)  & (df_proc["KA"]  <= 15) &
                (df_proc["FAT"] >= 0.5)& (df_proc["FAT"] <= 10) &
                (df_proc["SNF"] >= 3)  & (df_proc["SNF"] <= 15)
            ].copy()
            if len(df_proc) < 3:
                return pd.DataFrame()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df_proc[CLUST_FEATS])
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df_proc["Cluster"] = kmeans.fit_predict(scaled)
            # Urutkan: TS rendah → Kualitas Rendah, TS tinggi → Kualitas Tinggi
            mean_ts  = df_proc.groupby("Cluster")["TS"].mean().sort_values(ascending=True)
            rank_map = {old: new for new, old in enumerate(mean_ts.index)}
            df_proc["Cluster"] = df_proc["Cluster"].map(rank_map)
            return df_proc

        df_clust = jalankan_kmeans_custom(df_filtered)
        if df_clust.empty:
            st.warning("Data tidak cukup untuk clustering (minimal 3 baris valid).")
            st.stop()

        df_clust["Segmen"] = df_clust["Cluster"].map(CLUST_LABELS)
        st.success(f"Analisis berhasil! **{len(df_clust):,} data** · **{df_clust['NAMA_KELOMPOK'].nunique()} TPK**")

        cols_to_format = ["KA", "FAT", "SNF", "TS"]

        # ── Segmentasi TPK: assign tiap TPK ke segmen berdasarkan rata-rata TS ──
        # (bukan majority vote, karena cluster 2/Tinggi tersebar tipis di semua TPK)
        tpk_chars = (
            df_clust.groupby("NAMA_KELOMPOK")[cols_to_format]
            .mean().round(3).reset_index()
        )
        # Bagi 13 TPK ke 3 segmen merata berdasarkan ranking TS
        tpk_chars = tpk_chars.sort_values("TS", ascending=True).reset_index(drop=True)
        n_tpk     = len(tpk_chars)
        # Bagi sepertiga bawah=Rendah, tengah=Sedang, atas=Tinggi
        cut_low   = n_tpk // 3
        cut_mid   = cut_low * 2
        tpk_chars["Cluster_TPK"] = 0
        tpk_chars.loc[cut_low:cut_mid-1, "Cluster_TPK"] = 1
        tpk_chars.loc[cut_mid:,          "Cluster_TPK"] = 2
        tpk_chars["Segmen_TPK"] = tpk_chars["Cluster_TPK"].map(CLUST_LABELS)

        # ════════════════════════════════════════
        # BAGIAN 1 — GRAFIK RINGKASAN
        # ════════════════════════════════════════
        st.markdown("---")
        st.markdown(
            "<div style='background:#eaf4fb;border-left:4px solid #2980b9;"
            "padding:14px 18px;border-radius:6px;margin-bottom:12px;'>"
            "<div style='font-size:16px;margin-bottom:6px;'>"
            "🔵 <b>Metode Segmentasi:</b> <b>K-Means Clustering</b> (k=3)"
            "</div>"
            "<div style='font-size:15px;color:#2c3e50;'>"
            "Clustering dilakukan pada fitur fisik susu (KA, FAT, SNF, TS) "
            "dengan normalisasi <i>StandardScaler</i>. "
            "Jumlah cluster ditentukan berdasarkan <i>Elbow Method</i> dan <i>Silhouette Score</i>."
            "</div>"
            "</div>",
            unsafe_allow_html=True)
        st.markdown("### 📊 Ringkasan Segmentasi")

        mc1, mc2, mc3 = st.columns(3)
        for _col, _cid, _clr in zip([mc1, mc2, mc3], [0, 1, 2], [C_RED, C_ORANGE, C_GREEN]):
            _lbl  = CLUST_LABELS[_cid]
            _sub  = df_clust[df_clust["Cluster"] == _cid]
            _ntpk = tpk_chars[tpk_chars["Cluster_TPK"] == _cid].shape[0]
            _col.markdown(metric_card(
                _lbl, f"{len(_sub):,} data",
                f"{_ntpk} TPK  ·  TS avg: {_sub['TS'].mean():.2f}",
                color=_clr, border=_clr), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            fig_pie = px.pie(
                df_clust["Segmen"].value_counts().reset_index(),
                values="count", names="Segmen",
                color="Segmen", color_discrete_map=CLUST_COLORS,
                title="Distribusi Data per Segmen", hole=0.4)
            fig_pie.update_layout(height=320, margin=dict(t=40,b=10,l=0,r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            # Bar chart: jumlah TPK per segmen dari assignment berbasis TS ranking
            _bar_df = (
                tpk_chars.groupby("Segmen_TPK")["NAMA_KELOMPOK"]
                .count().reset_index()
                .rename(columns={"NAMA_KELOMPOK": "Jumlah Kelompok",
                                  "Segmen_TPK": "Segmen"})
            )
            # Pastikan urutan segmen konsisten
            _seg_order = [CLUST_LABELS[0], CLUST_LABELS[1], CLUST_LABELS[2]]
            _bar_df["Segmen"] = pd.Categorical(_bar_df["Segmen"], categories=_seg_order, ordered=True)
            _bar_df = _bar_df.sort_values("Segmen")
            fig_bar = px.bar(
                _bar_df, x="Segmen", y="Jumlah Kelompok",
                color="Segmen", color_discrete_map=CLUST_COLORS,
                title="Jumlah Kelompok per Segmen", text="Jumlah Kelompok")
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(height=320, margin=dict(t=40,b=60,l=0,r=0),
                plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                xaxis=dict(showgrid=False, tickangle=-10),
                yaxis=dict(gridcolor="#f0f0f0", range=[0, _bar_df["Jumlah Kelompok"].max() + 2]))
            st.plotly_chart(fig_bar, use_container_width=True)

        # ════════════════════════════════════════
        # BAGIAN 2 — TABEL AGREGAT
        # ════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 📋 Daftar Kelompok per Segmen")

        # Gabung dengan segmen TPK yang sudah diassign
        tpk_summary = tpk_chars[["NAMA_KELOMPOK", "Segmen_TPK"] + cols_to_format].copy()
        tpk_summary = tpk_summary.rename(columns={"Segmen_TPK": "Segmen"})
        tpk_summary = tpk_summary.sort_values(["Segmen", "TS"]).reset_index(drop=True)

        # Warna per baris: merah=Rendah, orange=Sedang, hijau=Tinggi
        def color_segmen_row(row):
            c = CLUST_COLORS.get(row["Segmen"], "#ffffff")
            return [f"background-color:{c}22"] * len(row)

        st.dataframe(
            tpk_summary.style
                .apply(color_segmen_row, axis=1)
                .background_gradient(cmap="RdYlGn_r", subset=["KA"])
                .background_gradient(cmap="Greens",   subset=["FAT", "SNF", "TS"])
                .format("{:.2f}", subset=cols_to_format),
            use_container_width=True)

        # ════════════════════════════════════════
        # BAGIAN 3 — SIDE-BY-SIDE PER SEGMEN
        # ════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🗂️ Kelompok per Segmen (Side-by-Side)")
        st.caption("Tiap TPK diassign ke segmen berdasarkan ranking rata-rata TS: sepertiga bawah=Rendah, tengah=Sedang, atas=Tinggi.")

        # Hitung vmin/vmax absolut dari SEMUA TPK supaya warna konsisten antar segmen
        _vmin = {c: tpk_chars[c].min() for c in cols_to_format}
        _vmax = {c: tpk_chars[c].max() for c in cols_to_format}

        col_r, col_s, col_t = st.columns(3)
        for _col_ui, _cid in zip([col_r, col_s, col_t], [0, 1, 2]):
            _lbl    = CLUST_LABELS[_cid]
            _subset = tpk_chars[tpk_chars["Cluster_TPK"] == _cid].copy()
            _n      = len(_subset)
            with _col_ui:
                st.markdown(
                    f"<div style='{BADGE_STYLE[_lbl]};padding:8px 14px;"
                    f"border-radius:8px;font-weight:700;font-size:13px;"
                    f"margin-bottom:8px'>{_lbl} — {_n} TPK</div>",
                    unsafe_allow_html=True)
                if _subset.empty:
                    st.caption("Tidak ada kelompok di segmen ini.")
                else:
                    _tbl = (
                        _subset[["NAMA_KELOMPOK"] + cols_to_format]
                        .rename(columns={"NAMA_KELOMPOK": "Kelompok"})
                        .set_index("Kelompok")
                        .sort_values("TS", ascending=(_cid == 0))
                    )
                    _styled = _tbl.style.format("{:.3f}")
                    # KA: merah=tinggi(buruk), hijau=rendah(bagus) — skala absolut
                    _styled = _styled.background_gradient(
                        cmap="RdYlGn_r", subset=["KA"],
                        vmin=_vmin["KA"], vmax=_vmax["KA"])
                    # FAT, SNF, TS: hijau=tinggi(bagus), putih=rendah — skala absolut
                    for _fc in ["FAT", "SNF", "TS"]:
                        _styled = _styled.background_gradient(
                            cmap="Greens", subset=[_fc],
                            vmin=_vmin[_fc], vmax=_vmax[_fc])
                    st.dataframe(_styled, use_container_width=True,
                                 height=min(60 + _n * 38, 420))

    else:
        st.info("Upload file CSV untuk memulai segmentasi TPK.")
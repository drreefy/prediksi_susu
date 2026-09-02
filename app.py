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
import joblib, json, os, re, sqlite3
from datetime import datetime, time as dtime
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
# CONSTANTS
# ════════════════════════════════════════════
FEAT_COLS = ["Netto","Temp.","PH","TS","SNF","FAT","Density","Durasi_Menit"]
TARGET    = "TPC"
DB_PATH   = "database.db"

C_BLUE   = "#4A90D9"
C_ORANGE = "#e67e22"
C_PURPLE = "#9b59b6"
C_RED    = "#e74c3c"
C_GREEN  = "#2ecc71"
C_GRAY   = "#adb5bd"

# ════════════════════════════════════════════
# DATABASE HELPERS
# ════════════════════════════════════════════
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=20.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS master_data (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            Tgl          TEXT,
            Nama         TEXT,
            NoPol        TEXT,
            Netto        REAL,
            Segel        TEXT,
            Temp         REAL,
            Appearance   TEXT,
            TDO          TEXT,
            PH           REAL,
            AT           TEXT,
            BTB          TEXT,
            CT           TEXT,
            Antibiotik   TEXT,
            TS           REAL,
            SNF          REAL,
            FAT          REAL,
            TPC          REAL,
            Density      REAL,
            Durasi_Menit REAL,
            Bulan        TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_tpk (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            NO            TEXT,
            KLP_SAMPLE    TEXT,
            KA            REAL,
            FAT           REAL,
            SNF           REAL,
            TS            REAL,
            NAMA_KELOMPOK TEXT,
            TANGGAL       TEXT,
            WAKTU         TEXT,
            TANGGAL_ASLI  TEXT,
            TAHUN         INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS durasi_harian (
            Tgl          TEXT PRIMARY KEY,
            Durasi_Menit REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rekomendasi_segmen (
            segmen       TEXT PRIMARY KEY,
            rekomendasi  TEXT,
            updated_by   TEXT,
            updated_at   TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_rekomendasi_segmen():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM rekomendasi_segmen", conn)
    conn.close()
    return df

def save_rekomendasi_segmen(segmen, rekomendasi, username):
    conn = get_conn()
    from datetime import datetime as _dtnow
    now = _dtnow.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO rekomendasi_segmen (segmen, rekomendasi, updated_by, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(segmen) DO UPDATE SET
            rekomendasi=excluded.rekomendasi,
            updated_by=excluded.updated_by,
            updated_at=excluded.updated_at
    """, (segmen, rekomendasi, username, now))
    conn.commit()
    conn.close()

@st.cache_data(ttl=30)
def load_master_data():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM master_data", conn)
    conn.close()
    if df.empty:
        return df
    df = df.rename(columns={"Temp": "Temp."})
    df["Tgl"] = pd.to_datetime(df["Tgl"], errors="coerce")
    df["Bulan_Str"] = df["Tgl"].dt.to_period("M").astype(str)
    df["Nama"]  = df["Nama"].fillna("")
    df["NoPol"] = df["NoPol"].fillna("")
    return df

@st.cache_data(ttl=30)
def load_tpk_data():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM data_tpk", conn)
    conn.close()
    if not df.empty:
        df = df.rename(columns={"KLP_SAMPLE": "KLP SAMPLE"})
        df["TANGGAL_ASLI"] = pd.to_datetime(df["TANGGAL_ASLI"], errors="coerce")
    return df

def db_row_count(table):
    conn = get_conn()
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return n

@st.cache_data
def jalankan_kmeans_custom(df_in):
    CLUST_FEATS  = ["KA", "FAT", "SNF", "TS"]
    df_proc = df_in[CLUST_FEATS + ["NAMA_KELOMPOK"]].dropna().copy()
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
    mean_ts  = df_proc.groupby("Cluster")["TS"].mean().sort_values(ascending=True)
    rank_map = {old: new for new, old in enumerate(mean_ts.index)}
    df_proc["Cluster"] = df_proc["Cluster"].map(rank_map)
    return df_proc


# ════════════════════════════════════════════
# DURASI EXTRACTION FROM KUD FILE
# ════════════════════════════════════════════
def extract_durasi_from_kud(df_raw, filename):
    depart_hour = 6 if "PAGI" in filename.upper() else 15
    depart = dtime(depart_hour, 0)

    anchor_row = None
    for r in range(df_raw.shape[0]):
        for c in range(df_raw.shape[1]):
            val = str(df_raw.iloc[r, c]).strip().upper()
            if "KEDATANGAN" in val:
                anchor_row = r
                break
        if anchor_row is not None:
            break

    if anchor_row is None:
        return None

    jam_col = None
    header_row = None
    for r in range(anchor_row, min(anchor_row + 6, df_raw.shape[0])):
        for c in range(df_raw.shape[1]):
            val = str(df_raw.iloc[r, c]).strip().upper()
            if val == "JAM":
                jam_col = c
                header_row = r
                break
        if jam_col is not None:
            break

    if jam_col is None:
        return None

    minutes_list = []
    for r in range(header_row + 1, df_raw.shape[0]):
        raw_val = df_raw.iloc[r, jam_col]
        if pd.isna(raw_val):
            continue
        val_str = str(raw_val).strip()
        if val_str in ["", "nan", "NaN"]:
            continue

        try:
            if isinstance(raw_val, dtime):
                t = raw_val
            elif isinstance(raw_val, datetime):
                t = raw_val.time()
            else:
                parts = val_str.replace(".", ":").split(":")
                h, m = int(parts[0]), int(parts[1])
                t = dtime(h, m)

            arr_mins  = t.hour * 60 + t.minute
            dep_mins  = depart.hour * 60 + depart.minute
            diff = arr_mins - dep_mins
            if diff > 0:
                minutes_list.append(diff)
        except Exception:
            continue

    if not minutes_list:
        return None
    return round(np.mean(minutes_list), 1)


def interpolate_durasi(bulan_str):
    conn = get_conn()
    df_d = pd.read_sql(
        "SELECT Durasi_Menit, Bulan FROM master_data WHERE Durasi_Menit IS NOT NULL", conn)
    conn.close()
    if df_d.empty:
        return 210.0
    same = df_d[df_d["Bulan"] == bulan_str]["Durasi_Menit"]
    if not same.empty:
        return round(same.mean(), 1)
    return round(df_d["Durasi_Menit"].mean(), 1)


# ════════════════════════════════════════════
# DATA ULTRA PARSER
# ════════════════════════════════════════════
def parse_ultra_file(file_bytes, durasi_override=None):
    df_raw = pd.read_excel(file_bytes, header=None)

    header_row = None
    for r in range(min(10, df_raw.shape[0])):
        for c in range(df_raw.shape[1]):
            val = str(df_raw.iloc[r, c]).strip().upper()
            if val == "TGL":
                header_row = r
                break
        if header_row is not None:
            break

    if header_row is None:
        return None

    file_bytes.seek(0)
    
    df = pd.read_excel(file_bytes, header=header_row)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(how="all")

    col_map = {
        "Tgl": "Tgl", "Nama": "Nama", "NoPol": "NoPol",
        "Netto": "Netto", "Segel": "Segel",
        "Temp.": "Temp", "Temp": "Temp",
        "Appearance": "Appearance", "TDO": "TDO",
        "PH": "PH", "AT": "AT", "BTB": "BTB", "CT": "CT",
        "Antibiotik": "Antibiotik",
        "TS": "TS", "SNF": "SNF", "FAT": "FAT",
        "TPC": "TPC", "Density": "Density",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df["Tgl"]   = pd.to_datetime(df["Tgl"], errors="coerce")
    df["Bulan"] = df["Tgl"].dt.to_period("M").astype(str)
    df["Tgl"]   = df["Tgl"].dt.strftime("%Y-%m-%d")

    conn = get_conn()
    df_durasi = pd.read_sql("SELECT Tgl, Durasi_Menit as Durasi_KUD FROM durasi_harian", conn)
    conn.close()
    
    df = df.merge(df_durasi, on="Tgl", how="left")

    if durasi_override is not None:
        df["Durasi_Menit"] = durasi_override
    else:
        df["Durasi_Menit"] = df["Durasi_KUD"].fillna(df["Bulan"].apply(interpolate_durasi))
    
    df = df.drop(columns=["Durasi_KUD"], errors="ignore")

    keep = ["Tgl","Nama","NoPol","Netto","Segel","Temp","Appearance","TDO",
            "PH","AT","BTB","CT","Antibiotik","TS","SNF","FAT","TPC","Density",
            "Durasi_Menit","Bulan"]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df = df[keep].dropna(subset=["Tgl"])
    return df


# ════════════════════════════════════════════
# DATA KUD PARSER (for data_tpk)
# ════════════════════════════════════════════
KOREKSI_GRUP = {
    'TPK ASEP CINCIN': 'TPK CICIN', 'TPK CINCIN': 'TPK CICIN',
    'TPK SUTIYANA': 'TPK SUTIANA', 'TPK AYI SAHRONI': 'TPK AYI',
    'SAMPLE TPK EPI': 'TPK EPI', 'SAMPLE TPK AYI': 'TPK AYI',
    'SAMPLE SAJANG': 'SAJANG', 'SAMPLE HASAN': 'TPK HASAN',
    'SAMPLE': 'SAMPEL INDIVIDU', 'SAMPLE SJM': 'SAMPEL INDIVIDU',
    'SEMPLAN PERORANAGN': 'SAMPEL INDIVIDU', 'SEMPELAN PEREKOR SUMI': 'SAMPEL INDIVIDU',
    'SAMPING': 'SAMPEL INDIVIDU', 'KELOMPOK C': 'SAMPEL INDIVIDU',
    'IWA': 'SAMPEL INDIVIDU', 'SAMPLE KLP IWA': 'SAMPEL INDIVIDU',
    'AWIT': 'SAMPEL INDIVIDU', 'SAMPLE AWIT': 'SAMPEL INDIVIDU',
    'WAWAN': 'SAMPEL INDIVIDU', 'SAMPLE KLP WAWAN': 'SAMPEL INDIVIDU',
}

def parse_kud_file(file_bytes, filename):
    match = re.search(r'(\d{2})[._](\d{2})[._](\d{4})\s*(PAGI|SORE)', filename, re.IGNORECASE)
    if not match:
        return None, None

    tanggal = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    waktu   = match.group(4).upper()

    try:
        df_raw = pd.read_excel(file_bytes, header=None)
    except Exception:
        return None, None

    durasi = extract_durasi_from_kud(df_raw, filename)
    data_gabungan = []
    hitung_kelompok_tanpa_nama = 0

    for r in range(df_raw.shape[0]):
        for c in range(df_raw.shape[1]):
            val_sel = str(df_raw.iloc[r, c]).strip().upper()
            if val_sel not in ['NO', 'NO.', 'NOMOR']:
                continue

            nama_kelompok = ""
            for step in range(1, 4):
                if r - step >= 0:
                    val_1 = df_raw.iloc[r - step, c]
                    val_2 = df_raw.iloc[r - step, c + 1] if c + 1 < df_raw.shape[1] else np.nan
                    pieces = []
                    if pd.notna(val_1) and str(val_1).strip().lower() not in ['nan','']:
                        pieces.append(str(val_1).strip())
                    if pd.notna(val_2) and str(val_2).strip().lower() not in ['nan','']:
                        pieces.append(str(val_2).strip())
                    gabung = " ".join(pieces)
                    if gabung and "DATA HASIL" not in gabung.upper():
                        nama_kelompok = gabung
                        break

            nama_clean = nama_kelompok.strip().upper()
            if re.match(r'^\d+', nama_clean):
                nama_clean = ""
            if nama_clean.startswith("KELOMPOK ") and len(nama_clean) > 9:
                nama_clean = nama_clean[9:].strip()
            if nama_clean in ['', 'NAN', 'NONE', 'KELOMPOK', 'NULL']:
                hitung_kelompok_tanpa_nama += 1
                nama_final = f"KELOMPOK {chr(64 + hitung_kelompok_tanpa_nama)}"
            else:
                nama_final = nama_clean
            if "JAM KEDATANGAN" in nama_final or "SAMPLING" in nama_final:
                continue

            for r_data in range(r + 1, df_raw.shape[0]):
                val_no = str(df_raw.iloc[r_data, c]).strip()
                if val_no in ['', 'nan', 'NaN', 'NONE', 'None'] or val_no.upper().startswith('TOTAL'):
                    break
                row_data = [str(df_raw.iloc[r_data, col]).strip()
                            for col in range(c, min(c + 6, df_raw.shape[1]))]
                while len(row_data) < 6:
                    row_data.append("")
                row_data += [nama_final, tanggal, waktu]
                data_gabungan.append(row_data)

    if not data_gabungan:
        return None, durasi

    cols = ['NO','KLP_SAMPLE','KA','FAT','SNF','TS','NAMA_KELOMPOK','TANGGAL','WAKTU']
    df = pd.DataFrame(data_gabungan, columns=cols)
    df = df[df['KLP_SAMPLE'].str.lower() != 'nan']
    df = df[df['KLP_SAMPLE'] != '']
    df = df[~df['KLP_SAMPLE'].astype(str).str.upper().str.contains('SAMPLE', na=False)]
    df['NAMA_KELOMPOK'] = df['NAMA_KELOMPOK'].replace(r'\s+', ' ', regex=True)
    df['NAMA_KELOMPOK'] = df['NAMA_KELOMPOK'].replace(KOREKSI_GRUP)

    for col in ['KA','FAT','SNF','TS']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','.'), errors='coerce')

    df['TANGGAL_ASLI'] = pd.to_datetime(df['TANGGAL'], format='%d.%m.%Y', errors='coerce')
    df = df.dropna(subset=['TANGGAL_ASLI', 'KA', 'FAT', 'SNF', 'TS'])
    df['TAHUN'] = df['TANGGAL_ASLI'].dt.year.astype(int)
    df['TANGGAL_ASLI'] = df['TANGGAL_ASLI'].dt.strftime("%Y-%m-%d")

    return df, durasi


# ════════════════════════════════════════════
# DB INSERT HELPERS
# ════════════════════════════════════════════
def insert_master_data(df):
    with get_conn() as conn:
        df.to_sql("master_data", conn, if_exists="append", index=False)
    load_master_data.clear()

def insert_tpk_data(df):
    with get_conn() as conn:
        df.to_sql("data_tpk", conn, if_exists="append", index=False)
    load_tpk_data.clear()


# ════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════
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
        source = ("⚠️ model.pkl tidak kompatibel — model dilatih ulang dari data"
                  if os.path.exists(MODEL_PATH)
                  else "⚙️ model.pkl tidak ditemukan — model dilatih ulang dari data")
    d      = df[feat_names + [TARGET]].dropna()
    X      = d[feat_names]
    y      = np.log1p(d[TARGET])
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = mdl.predict(X_te)
    return mdl, feat_names, X_te, y_te, y_pred, source


def metric_card(label, value, sub="", color="#212529", border=None):
    border_style = f"border:2px solid {border};" if border else ""
    return f"""<div class='metric-card' style='{border_style}'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value' style='color:{color}'>{value}</div>
        <div class='metric-sub'>{sub}</div>
    </div>"""


# ════════════════════════════════════════════
# LOGIN
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
        st.markdown("<div style='background:#f8f9fa;padding:2rem;border-radius:12px;border:1px solid #e9ecef'>",
                    unsafe_allow_html=True)
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
    🔑 Admin: akses penuh &nbsp;|&nbsp; Petugas: akses terbatas
    </div>""", unsafe_allow_html=True)

if not st.session_state.get("logged_in"):
    login_page()
    st.stop()

ROLE = st.session_state["role"]

# ════════════════════════════════════════════
# LANDING PAGE
# ════════════════════════════════════════════
if "section" not in st.session_state:
    st.session_state["section"] = None

if st.session_state["section"] is None:
    st.markdown("""<style>
    [data-testid="stSidebar"]{display:none}
    div[data-testid="column"] div.stButton > button {
        background:white;border-radius:16px;height:260px;width:100%;
        border:2px solid #dee2e6;padding:2rem 1.5rem;
        font-size:13px;color:#6c757d;
        white-space:normal;text-align:center;
        box-shadow:0 2px 12px rgba(0,0,0,0.07);
        transition:all 0.2s ease;
    }
    div[data-testid="column"] div.stButton > button:hover {
        transform:translateY(-4px);
        box-shadow:0 8px 24px rgba(0,0,0,0.13);
    }
    </style>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align:center;padding:3rem 0 2rem'>
        <div style='font-size:72px'>🥛</div>
        <h2 style='margin:0.5rem 0;line-height:1.3;font-size:26px'>
            Sistem Monitoring Kualitas Susu<br>KUD Sarwa Mukti Cisarua
        </h2>
        <p style='color:#6c757d;font-size:15px;margin-top:0.5rem'>
            Selamat datang, <b>{st.session_state['username']}</b>. Pilih menu untuk melanjutkan.
        </p>
    </div>""", unsafe_allow_html=True)

    if ROLE == "admin":
        cards = [
            ("🗄️ Data Historis", "🗄️", "Data Historis",
             "Upload data KUD dan Data Ultra,\nlihat dan kelola isi database.", "#2ecc71"),
            ("📊 Analisis TPC",  "📊", "Analisis TPC",
             "Overview, statistika deskriptif, eksplorasi data,\nperforma model, dan simulasi prediksi TPC.", "#4A90D9"),
            ("🔬 Segmentasi TPK","🔬", "Segmentasi TPK",
             "Segmentasi kelompok peternak berdasarkan\nkualitas susu menggunakan K-Means Clustering.", "#e67e22"),
        ]
    else:
        cards = [
            ("📊 Analisis TPC",  "📊", "Analisis TPC",
             "Overview dan simulasi prediksi TPC\nberdasarkan data susu.", "#4A90D9"),
            ("🔬 Segmentasi TPK","🔬", "Segmentasi TPK",
             "Segmentasi kelompok peternak berdasarkan\nkualitas susu menggunakan K-Means Clustering.", "#e67e22"),
        ]

    card_styles = "".join([
        f"div[data-testid='column']:nth-child({i+1}) div.stButton > button {{"
        f"border-color:{color} !important;}}"
        f"div[data-testid='column']:nth-child({i+1}) div.stButton > button:hover {{"
        f"border-color:{color} !important;background:{color}0d !important;}}"
        for i, (*_, color) in enumerate(cards)
    ])
    st.markdown(f"<style>{card_styles}</style>", unsafe_allow_html=True)

    cols = st.columns(len(cards))
    for col, (key, icon, title, desc, color) in zip(cols, cards):
        with col:
            if st.button(f"{icon}\n\n**{title}**\n\n{desc}", key=f"card_{key}", use_container_width=True):
                st.session_state["section"] = key
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    c_lo = st.columns([3,1,3])[1]
    with c_lo:
        if st.button("🚪 Logout", key="landing_logout", use_container_width=True):
            for k in ["logged_in","username","role","section"]:
                st.session_state.pop(k, None)
            st.rerun()
    st.stop()

section = st.session_state["section"]

# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🥛 Sistem Monitoring Kualitas Susu\nKUD Sarwa Mukti Cisarua")
    st.markdown("---")
    st.markdown(f"👤 **{st.session_state['username']}** "
                f"({'Admin' if ROLE == 'admin' else 'Petugas Koperasi'})")
    col_home, col_out = st.columns(2)
    with col_home:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state["section"] = None
            st.rerun()
    with col_out:
        if st.button("🚪 Logout", use_container_width=True):
            for k in ["logged_in","username","role","section"]:
                st.session_state.pop(k, None)
            st.rerun()
    st.markdown("---")

    bulan_range = ("", "")
    sel_kud     = "Semua"
    sel_nopol   = "Semua"
    sel_tgl_dari = None
    sel_tgl_sampai = None
    page        = None

    if section == "📊 Analisis TPC":
        df_raw   = load_master_data()
        has_data = not df_raw.empty

        if has_data:
            months_all  = sorted(df_raw["Bulan_Str"].dropna().unique())
            st.markdown("### Filter")
            if len(months_all) > 1:
                bulan_range = st.select_slider("Rentang Bulan", options=months_all,
                                               value=(months_all[0], months_all[-1]))
            else:
                st.info(f"📅 Data bulan: **{months_all[0]}**")
                bulan_range = (months_all[0], months_all[0])

            with st.expander("📅 Filter Tanggal Spesifik"):
                import datetime as _dt
                _tgl_s = pd.to_datetime(df_raw["Tgl"], errors="coerce").dropna()
                if not _tgl_s.empty:
                    _tgl_min, _tgl_max = _tgl_s.min().date(), _tgl_s.max().date()
                    sel_tgl_dari   = st.date_input("Dari Tanggal",   value=_tgl_min, min_value=_tgl_min, max_value=_tgl_max, key="tgl_dari_tpc")
                    sel_tgl_sampai = st.date_input("Sampai Tanggal", value=_tgl_max, min_value=_tgl_min, max_value=_tgl_max, key="tgl_sampai_tpc")
                else:
                    st.caption("Data tanggal tidak tersedia.")

            if ROLE == "admin":
                kud_opts   = ["Semua"] + sorted(df_raw["Nama"].dropna().unique().tolist())
                nopol_opts = ["Semua"] + sorted(df_raw["NoPol"].dropna().unique().tolist())
                sel_kud   = st.selectbox("KUD",   kud_opts)
                sel_nopol = st.selectbox("NoPol", nopol_opts)
            st.markdown("---")

        if ROLE == "admin":
            page_opts = ["📊 Overview","📈 Statistika Deskriptif","🔍 Eksplorasi Data",
                         "🤖 Model Performance","🧪 Simulasi Prediksi"]
        else:
            page_opts = ["📊 Overview","🧪 Simulasi Prediksi"]
        page = st.radio("Halaman", page_opts)

    elif section == "🔬 Segmentasi TPK":
        df_tpk_sb = load_tpk_data()
        if not df_tpk_sb.empty:
            st.markdown("### Filter")
            with st.expander("📅 Filter Tanggal Spesifik"):
                import datetime as _dt
                _tgl_s = pd.to_datetime(df_tpk_sb["TANGGAL_ASLI"], errors="coerce").dropna()
                if not _tgl_s.empty:
                    _tgl_min, _tgl_max = _tgl_s.min().date(), _tgl_s.max().date()
                    sel_tgl_dari   = st.date_input("Dari Tanggal",   value=_tgl_min, min_value=_tgl_min, max_value=_tgl_max, key="tgl_dari_tpk")
                    sel_tgl_sampai = st.date_input("Sampai Tanggal", value=_tgl_max, min_value=_tgl_min, max_value=_tgl_max, key="tgl_sampai_tpk")
                else:
                    st.caption("Data tanggal tidak tersedia.")
            st.markdown("---")

# ════════════════════════════════════════════
# SECTION: ANALISIS TPC
# ════════════════════════════════════════════
if section == "📊 Analisis TPC":
    df_raw = load_master_data()

    if df_raw.empty:
        st.markdown("""
        <div style='text-align:center;padding:6rem 0'>
            <div style='font-size:80px'>🥛</div>
            <h2>Belum Ada Data</h2>
            <p style='color:#6c757d'>Upload data melalui halaman <b>Data Historis</b> untuk memulai.</p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    df = df_raw[(df_raw["Bulan_Str"] >= bulan_range[0]) & (df_raw["Bulan_Str"] <= bulan_range[1])].copy()
    if sel_kud   != "Semua": df = df[df["Nama"]  == sel_kud]
    if sel_nopol != "Semua": df = df[df["NoPol"] == sel_nopol]
    
    if sel_tgl_dari is not None and sel_tgl_sampai is not None:
        _tgl_col = pd.to_datetime(df["Tgl"], errors="coerce")
        import datetime as _dt
        df = df[(_tgl_col >= pd.Timestamp(sel_tgl_dari)) &
                (_tgl_col <= pd.Timestamp(sel_tgl_sampai))].copy()

    months_f = sorted(df["Bulan_Str"].dropna().unique())
    monthly = (
        df.groupby("Bulan_Str")[TARGET]
        .agg(Jumlah="count", Mean="mean", Median="median", Std="std",
             Min="min", Max="max",
             Q1=lambda x: x.quantile(0.25), Q3=lambda x: x.quantile(0.75))
        .reindex(months_f).round(4)
    )

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
        st.dataframe(monthly, use_container_width=True)

# ════════════════════════════════════════════
# PAGE 3 — EKSPLORASI DATA
# ════════════════════════════════════════════
    elif page == "🔍 Eksplorasi Data":
        st.markdown("## 🔍 Eksplorasi Data")

        tahun_opts_ex = ["Semua"] + sorted(pd.to_datetime(df["Tgl"], errors="coerce").dt.year.dropna().unique().astype(int).tolist())
        sel_tahun_ex  = st.selectbox("Filter Tahun", tahun_opts_ex, index=0, key="eksplorasi_tahun")
        df_ex = df[pd.to_datetime(df["Tgl"], errors="coerce").dt.year == sel_tahun_ex].copy() if sel_tahun_ex != "Semua" else df.copy()
        st.caption(f"{len(df_ex):,} data ditampilkan" + (f" · Tahun {sel_tahun_ex}" if sel_tahun_ex != "Semua" else " · Semua tahun"))

        num_cols = FEAT_COLS + [TARGET]
        cl, cr = st.columns(2)
        x_ax = cl.selectbox("Sumbu X", num_cols, index=0)
        y_ax = cr.selectbox("Sumbu Y", num_cols, index=num_cols.index(TARGET))

        df_ex["Tahun"] = pd.to_datetime(df_ex["Tgl"], errors="coerce").dt.year.astype(str)
        cb_opts = ["Tahun", "Bulan_Str", "Nama", "NoPol"]
        color_by = st.selectbox("Warna berdasarkan", cb_opts)

        BRIGHT_COLORS = ["#e6000a","#007bff","#00b300","#ff8c00","#9400d3","#00bcd4","#ff1493","#8B4513"]
        fig_s = px.scatter(df_ex, x=x_ax, y=y_ax, color=color_by,
            hover_data=["Bulan_Str","Nama","NoPol"],
            color_discrete_sequence=BRIGHT_COLORS)
        fig_s.update_traces(marker=dict(size=8, opacity=1,
                                        line=dict(width=0.3, color="white")))
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
        FEAT_DEFAULT = {"Durasi_Menit": 210.0}

        for i, feat in enumerate(feat_names):
            col  = cols3[i % 4]
            fmin  = float(ref[feat].min())
            fmax  = float(ref[feat].max())
            fmean = float(ref[feat].mean())
            step  = round((fmax - fmin) / 100, 2) if fmax != fmin else 0.01
            lbl   = feat_meta.get(feat, (feat,""))[0]
            default_val = FEAT_DEFAULT.get(feat, round(fmean, 2))
            inputs[feat] = col.number_input(lbl, value=default_val, step=step, format="%.2f")

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

            THRESHOLD = 1  
            tpc_safe  = pred_tpc <= THRESHOLD

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

            try:
                explainer   = shap.TreeExplainer(mdl)
                shap_vals   = explainer.shap_values(inp_df)[0]
                shap_dict   = dict(zip(feat_names, shap_vals))
                shap_sorted = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

                total_abs = sum(abs(sv) for _, sv in shap_sorted) or 1

                fitur_diluar_sni = set()
                for feat, val in inputs.items():
                    bounds = FEAT_BOUNDS.get(feat, {})
                    lo, hi = bounds.get("min"), bounds.get("max")
                    if (lo is not None and val < lo) or (hi is not None and val > hi):
                        fitur_diluar_sni.add(feat)

                if tpc_safe:
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

            st.markdown("---")
            st.markdown("### 💡 Rekomendasi Penanganan")

            if tpc_safe and not ada_peringatan:
                st.success("Kondisi susu optimal. Semua parameter memenuhi standar SNI.")
            else:
                # Ikon & warna khas per variabel
                FITUR_STYLE = {
                    "Durasi_Menit": {"icon": "⏱️", "color": "#e67e22"},
                    "Temp.":        {"icon": "🌡️", "color": "#e74c3c"},
                    "PH":           {"icon": "🧪", "color": "#9b59b6"},
                    "FAT":          {"icon": "🧈", "color": "#3498db"},
                    "SNF":          {"icon": "🌾", "color": "#27ae60"},
                    "TS":           {"icon": "🐄", "color": "#16a085"},
                }
                REKOMENDASI_FITUR = {
                    "Durasi_Menit": {"high": "Persingkat durasi penanganan susu hingga maksimal 4 jam (240 menit) sejak pemerahan. Semakin lama susu dibiarkan pada suhu ruang, semakin cepat bakteri berkembang biak dan meningkatkan nilai TPC."},
                    "Temp.": {
                        "high": "Turunkan suhu penyimpanan susu sesuai standar (1–4 °C). Suhu tinggi mempercepat pertumbuhan mikroba sehingga langsung berdampak pada kenaikan TPC.",
                        "low":  "Pastikan suhu penyimpanan tidak terlalu rendah agar kondisi susu tetap normal dan tidak merusak komponen susu.",
                    },
                    "PH": {
                        "high": "Periksa kebersihan alat pemerahan dan wadah penyimpanan. pH yang terlalu basa dapat mengindikasikan adanya kontaminan atau residu sabun/detergen pada peralatan.",
                        "low":  "pH yang terlalu asam menandakan susu sudah mengalami fermentasi awal akibat aktivitas bakteri. Percepat pendinginan dan persingkat waktu penanganan.",
                    },
                    "FAT": {"low": "Kadar lemak rendah dapat mengindikasikan sapi dalam kondisi stres atau kekurangan pakan berkualitas. Perbaiki manajemen pakan dengan komposisi yang seimbang."},
                    "SNF": {"low": "SNF rendah menunjukkan kandungan padatan non-lemak di bawah standar. Pastikan sapi mendapat nutrisi yang cukup, terutama protein dan mineral dalam ransum pakan."},
                    "TS":  {"low": "Total Solid rendah mengindikasikan susu terlalu encer. Pastikan tidak terjadi pengenceran susu dan periksa kondisi kesehatan sapi."},
                }

                rek_items = []

                if not tpc_safe and "shap_sorted" in dir() and "total_abs" in dir():
                    try:
                        for f, sv in shap_sorted:
                            pct = abs(sv) / total_abs * 100
                            if sv > 0 and pct >= 2:
                                rek = REKOMENDASI_FITUR.get(f, {}).get("high") or REKOMENDASI_FITUR.get(f, {}).get("low")
                                if rek and f not in [r["feat"] for r in rek_items]:
                                    rek_items.append({"feat": f, "label": feat_meta.get(f,(f,""))[0],
                                                       "pct": pct, "rek": rek})
                    except Exception:
                        pass

                for direction, feat, lbl, val, lo, hi, unit, msg in peringatan:
                    if feat not in [r["feat"] for r in rek_items]:
                        rek = REKOMENDASI_FITUR.get(feat, {}).get(direction)
                        if rek:
                            rek_items.append({"feat": feat, "label": lbl, "pct": None, "rek": rek})

                if not rek_items and not tpc_safe:
                    rek_items.append({"feat": "TPC", "label": "TPC Melebihi Batas SNI", "pct": None,
                                       "rek": "Segera lakukan pendinginan cepat dan sanitasi menyeluruh terhadap wadah penampungan susu."})

                if rek_items:
                    st.caption("Rekomendasi disusun berdasarkan variabel yang berpengaruh terhadap prediksi TPC "
                               "dan parameter yang menyimpang dari standar SNI.")
                    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)

                    n_cols_rek = 2
                    rows_rek   = [rek_items[i:i+n_cols_rek] for i in range(0, len(rek_items), n_cols_rek)]
                    for row_rek in rows_rek:
                        cols_rek = st.columns(n_cols_rek)
                        for ci, item in enumerate(row_rek):
                            style = FITUR_STYLE.get(item["feat"], {"icon": "🚨", "color": "#c0392b"})
                            icon, warna = style["icon"], style["color"]
                            pct_badge = (f"<span style='background:{warna}18;color:{warna};font-size:13px;"
                                         f"font-weight:700;padding:4px 12px;border-radius:20px;margin-left:10px'>"
                                         f"{item['pct']:.1f}%</span>"
                                         if item["pct"] is not None else "")
                            with cols_rek[ci]:
                                st.markdown(
                                    f"<div style='background:white;border:1px solid #eee;"
                                    f"border-top:4px solid {warna};border-radius:10px;"
                                    f"padding:20px 22px;margin-bottom:16px;min-height:200px;"
                                    f"box-shadow:0 2px 8px rgba(0,0,0,0.06);"
                                    f"display:flex;flex-direction:column;'>"
                                    f"<div style='font-size:32px;margin-bottom:8px'>{icon}</div>"
                                    f"<div style='font-weight:700;font-size:19px;color:#222;margin-bottom:10px'>"
                                    f"{item['label']}{pct_badge}</div>"
                                    f"<div style='color:#495057;font-size:16px;line-height:1.65'>"
                                    f"{item['rek']}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True)

# ════════════════════════════════════════════
# SECTION: DATA HISTORIS (admin only)
# ════════════════════════════════════════════
elif section == "🗄️ Data Historis":
    st.markdown("## 🗄️ Data Historis")
    st.caption("Kelola database data susu — lihat, tambah, dan monitor data master dan data TPK.")
    
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    n_master = db_row_count("master_data")
    n_tpk    = db_row_count("data_tpk")
    c1, c2   = st.columns(2)
    c1.markdown(metric_card("Master Data (Analisis TPC)", f"{n_master:,} baris",
                             "Data Ultra — digunakan untuk prediksi & analisis",
                             color=C_BLUE, border=C_BLUE), unsafe_allow_html=True)
    c2.markdown(metric_card("Data TPK (Segmentasi)", f"{n_tpk:,} baris",
                             "Data KUD — digunakan untuk segmentasi TPK",
                             color=C_GREEN, border=C_GREEN), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ➕ Upload Data Baru")

    col_up, col_type = st.columns([2, 1])
    with col_up:
        up_files = st.file_uploader(
            "Upload file Excel", 
            type=["xlsx"], 
            accept_multiple_files=True, 
            key=f"hist_upload_{st.session_state['uploader_key']}"
        )
    with col_type:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        data_type = st.selectbox("Jenis Data", ["Data Ultra", "Data KUD"], key="hist_type")

    if up_files:
        st.markdown("---")
        st.info(f"📂 **{len(up_files)} file** siap diproses sebagai **{data_type}**.")

        first_file = up_files[0]
        st.markdown(f"**Preview file pertama ({first_file.name}) - 5 baris pertama:**")
        try:
            df_prev = pd.read_excel(first_file, header=None)
            hrow = None
            for r in range(min(10, df_prev.shape[0])):
                for c_i in range(df_prev.shape[1]):
                    if str(df_prev.iloc[r, c_i]).strip().upper() == "TGL":
                        hrow = r
                        break
                if hrow is not None:
                    break
            if hrow is not None:
                df_show = pd.read_excel(first_file, header=hrow).dropna(how="all").head(5)
                st.dataframe(df_show, use_container_width=True)
            first_file.seek(0)
        except Exception as e:
            st.warning(f"Preview gagal: {e}")
            first_file.seek(0)

        if st.button("✅ Proses & Simpan Semua", type="primary", key="save_all"):
            with st.spinner(f"Memproses {len(up_files)} file..."):
                total_rows = 0
                
                def get_file_date(f):
                    match = re.search(r'(\d{2})[._](\d{2})[._](\d{4})', f.name)
                    if match:
                        return datetime.strptime(f"{match.group(3)}-{match.group(2)}-{match.group(1)}", "%Y-%m-%d")
                    return datetime.min 

                up_files = sorted(up_files, key=get_file_date)
                
                for file in up_files:
                    fname = file.name
                    file.seek(0)

                    if data_type == "Data Ultra":
                        df_parsed = parse_ultra_file(file)
                        if df_parsed is None or df_parsed.empty:
                            st.error(f"❌ {fname}: Gagal parsing file. Pastikan format sesuai Data Ultra.")
                        else:
                            df_parsed = df_parsed.sort_values("Tgl", ascending=True)
                            
                            insert_master_data(df_parsed)
                            total_rows += len(df_parsed)
                            st.success(f"✅ {fname}: {len(df_parsed):,} baris berhasil disimpan!")

                    else: 
                        has_waktu = bool(re.search(r'(PAGI|SORE)', fname, re.IGNORECASE))
                        if not has_waktu:
                            st.warning(f"⚠️ {fname}: Nama file tidak mengandung PAGI atau SORE.")

                        df_tpk_parsed, durasi = parse_kud_file(file, fname)

                        if df_tpk_parsed is None or df_tpk_parsed.empty:
                            st.error(f"❌ {fname}: Gagal mengekstrak data TPK. Pastikan format sesuai Data KUD.")
                        else:
                            df_tpk_parsed = df_tpk_parsed.sort_values("TANGGAL_ASLI", ascending=True)
                            
                            if durasi is not None:
                                tgl_kud = df_tpk_parsed["TANGGAL_ASLI"].iloc[0]
                                with get_conn() as conn:
                                    conn.execute("INSERT OR REPLACE INTO durasi_harian (Tgl, Durasi_Menit) VALUES (?, ?)", (tgl_kud, durasi))
                                st.success(f"⏱️ {fname}: Durasi ({durasi:.1f} menit) disimpan untuk tanggal {tgl_kud}!")
                            else:
                                st.warning(f"⚠️ {fname}: Jam Kedatangan tidak ditemukan.")

                            insert_tpk_data(df_tpk_parsed)
                            total_rows += len(df_tpk_parsed)
                            st.success(f"✅ {fname}: {len(df_tpk_parsed):,} baris berhasil disimpan!")

                if total_rows > 0:
                    st.session_state["data_processed"] = True 
                    st.session_state["uploader_key"] += 1
                    st.rerun()

                if st.session_state.get("data_processed", False):
                    st.success("Data berhasil diunggah!")
                    st.session_state["data_processed"] = False 

    st.markdown("---")

    tab_master, tab_tpk = st.tabs(["📋 Master Data", "📋 Data TPK"])

    with tab_master:
        df_m = load_master_data()
        if df_m.empty:
            st.info("Belum ada data master. Upload Data Ultra melalui form di atas.")
        else:
            _bstr = df_m["Bulan_Str"].dropna()
            _bstr = _bstr[(_bstr != "") & (_bstr != "NaT")]
            _caption = (f"{len(df_m):,} baris · Periode: {_bstr.min()} — {_bstr.max()}"
                        if not _bstr.empty else f"{len(df_m):,} baris")
            st.caption(_caption)
            
            _dm = (df_m.drop(columns=["id","Bulan_Str"], errors="ignore")
                       .sort_values("Tgl", ascending=False, na_position="last")
                       .head(100))
            
            st.table(_dm)

    with tab_tpk:
        df_t = load_tpk_data()
        if df_t.empty:
            st.info("Belum ada data TPK.")
        else:
            try:
                _tahun = sorted(df_t["TAHUN"].dropna().astype(int).unique().tolist())
            except Exception:
                _tahun = []
            st.caption(f"{len(df_t):,} baris · Tahun: {_tahun}")
            
            _dt = (df_t.drop(columns=["id"], errors="ignore")
                       .sort_values("TANGGAL_ASLI", ascending=False, na_position="last")
                       .head(100))
            
            st.table(_dt)


# ════════════════════════════════════════════
# SECTION: SEGMENTASI TPK
# ════════════════════════════════════════════
elif section == "🔬 Segmentasi TPK":
    st.markdown("## 🔬 Segmentasi TPK")
    st.caption("Segmentasi TPK berdasarkan karakteristik fisik susu (KA, FAT, SNF, TS) menggunakan K-Means Clustering.")

    df_clust_raw = load_tpk_data()

    if df_clust_raw.empty:
        st.markdown("""
        <div style='text-align:center;padding:4rem 0'>
            <div style='font-size:64px'>🐄</div>
            <h3>Belum Ada Data TPK</h3>
            <p style='color:#6c757d'>Upload Data KUD melalui halaman <b>Data Historis</b> untuk mengisi database TPK.</p>
        </div>""", unsafe_allow_html=True)
        st.stop()

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

    cols_to_format = CLUST_FEATS

    for col in CLUST_FEATS:
        df_clust_raw[col] = pd.to_numeric(df_clust_raw[col], errors="coerce")
    df_clust_raw = df_clust_raw[df_clust_raw["NAMA_KELOMPOK"].astype(str).str.upper() != "SAMPEL INDIVIDU"].copy()

    # --- Eksekusi Filter Tanggal dari Sidebar ---
    if sel_tgl_dari is not None and sel_tgl_sampai is not None:
        _tgl_col = pd.to_datetime(df_clust_raw["TANGGAL_ASLI"], errors="coerce")
        df_filtered = df_clust_raw[(_tgl_col >= pd.Timestamp(sel_tgl_dari)) &
                                   (_tgl_col <= pd.Timestamp(sel_tgl_sampai))].copy()
        st.caption(f"📅 Memfilter data dari **{sel_tgl_dari.strftime('%d %b %Y')}** s/d **{sel_tgl_sampai.strftime('%d %b %Y')}**")
    else:
        df_filtered = df_clust_raw.copy()
    
    df_clust = jalankan_kmeans_custom(df_filtered)
    if df_clust.empty:
        st.warning("⚠️ Data tidak cukup untuk melakukan segmentasi (minimal 3 baris valid) pada rentang tanggal yang dipilih.")
        st.stop()

    df_clust["Segmen"] = df_clust["Cluster"].map(CLUST_LABELS)
    st.success(f"Analisis berhasil! **{len(df_clust):,} data** · **{df_clust['NAMA_KELOMPOK'].nunique()} TPK**")

    tpk_chars = (
        df_clust.groupby("NAMA_KELOMPOK")[cols_to_format]
        .mean().round(3).reset_index()
    )
    tpk_chars = tpk_chars.sort_values("TS", ascending=True).reset_index(drop=True)
    n_tpk_total = len(tpk_chars)
    cut_low  = n_tpk_total // 3
    cut_mid  = cut_low * 2
    tpk_chars["Cluster_TPK"] = 0
    tpk_chars.loc[cut_low:cut_mid-1, "Cluster_TPK"] = 1
    tpk_chars.loc[cut_mid:,          "Cluster_TPK"] = 2
    tpk_chars["Segmen_TPK"] = tpk_chars["Cluster_TPK"].map(CLUST_LABELS)

    # ════════════════════════
    # BAGIAN 1 — RINGKASAN
    # ════════════════════════
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
    for col, cid, clr in zip([mc1, mc2, mc3], [0, 1, 2], [C_RED, C_ORANGE, C_GREEN]):
        lbl  = CLUST_LABELS[cid]
        sub  = df_clust[df_clust["Cluster"] == cid]
        ntpk = tpk_chars[tpk_chars["Cluster_TPK"] == cid].shape[0]
        col.markdown(metric_card(
            lbl, f"{len(sub):,} data",
            f"{ntpk} TPK  ·  TS avg: {sub['TS'].mean():.2f}",
            color=clr, border=clr), unsafe_allow_html=True)

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
        bar_df = (tpk_chars.groupby("Segmen_TPK")["NAMA_KELOMPOK"]
                  .count().reset_index()
                  .rename(columns={"NAMA_KELOMPOK":"Jumlah Kelompok","Segmen_TPK":"Segmen"}))
        seg_order = [CLUST_LABELS[0], CLUST_LABELS[1], CLUST_LABELS[2]]
        bar_df["Segmen"] = pd.Categorical(bar_df["Segmen"], categories=seg_order, ordered=True)
        bar_df = bar_df.sort_values("Segmen")
        fig_bar = px.bar(bar_df, x="Segmen", y="Jumlah Kelompok",
                         color="Segmen", color_discrete_map=CLUST_COLORS,
                         title="Jumlah Kelompok per Segmen", text="Jumlah Kelompok")
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=320, margin=dict(t=40,b=60,l=0,r=0),
                               plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                               xaxis=dict(showgrid=False, tickangle=-10),
                               yaxis=dict(gridcolor="#f0f0f0",
                                          range=[0, bar_df["Jumlah Kelompok"].max()+2]))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ════════════════════════
    # BAGIAN 2 — TABEL AGREGAT (DENGAN WARNA GRADASI AMAN HTML)
    # ════════════════════════
    st.markdown("---")
    st.markdown("### 📋 Daftar Kelompok per Segmen")
    tpk_summary = tpk_chars[["NAMA_KELOMPOK","Segmen_TPK"] + cols_to_format].copy()
    tpk_summary = tpk_summary.rename(columns={"Segmen_TPK":"Segmen"})
    tpk_summary = tpk_summary.sort_values(["Segmen","TS"]).reset_index(drop=True)

    def get_cell_color(col_name, val):
        if pd.isna(val): return "#ffffff", "#000000"
        if col_name == "KA":
            if val > 6.5: return "#8B0000", "#ffffff"
            elif val > 6.0: return "#d4ac0d", "#000000"
            elif val > 5.5: return "#f9e79f", "#000000"
            else: return "#e8f8f5", "#000000"
        else:
            if col_name == "FAT": min_v, max_v = 3.0, 4.0
            elif col_name == "SNF": min_v, max_v = 7.5, 8.5
            else: min_v, max_v = 11.0, 12.0
            
            ratio = max(0.0, min(1.0, (val - min_v) / (max_v - min_v + 1e-5)))
            if ratio > 0.8: return "#145a32", "#ffffff"
            elif ratio > 0.6: return "#1e8449", "#ffffff"
            elif ratio > 0.4: return "#52be80", "#000000"
            elif ratio > 0.2: return "#a9dfbf", "#000000"
            else: return "#e8f8f5", "#000000"

    def render_colored_table_full(df):
        html = "<div style='overflow-x:auto;'><table style='width:100%; border-collapse:collapse; background:white; font-size:14px;'>"
        html += "<tr style='background:#f1f3f5; border-bottom:2px solid #dee2e6;'>"
        for col in df.columns:
            html += f"<th style='padding:10px; text-align:left; color:#212529;'>{col}</th>"
        html += "</tr>"
        
        for _, row in df.iterrows():
            seg_val = str(row["Segmen"])
            bg_row = "#ffffff"
            if "Rendah" in seg_val: bg_row = "#fdf2f2"
            elif "Sedang" in seg_val: bg_row = "#fffdf0"
            elif "Tinggi" in seg_val: bg_row = "#f2fdf5"
            
            html += f"<tr style='border-bottom:1px solid #e9ecef; background:{bg_row};'>"
            for col in df.columns:
                val = row[col]
                if col in ["KA", "FAT", "SNF", "TS"]:
                    bg_c, text_c = get_cell_color(col, val)
                    val_str = f"{val:.2f}" if isinstance(val, (float, np.floating)) else str(val)
                    html += f"<td style='padding:10px; background:{bg_c}; color:{text_c}; text-align:center;'>{val_str}</td>"
                else:
                    val_str = str(val)
                    html += f"<td style='padding:10px; color:#333; font-weight:500;'>{val_str}</td>"
            html += "</tr>"
        html += "</table></div>"
        return html

    st.markdown(render_colored_table_full(tpk_summary), unsafe_allow_html=True)

    # ════════════════════════
    # BAGIAN 3 — SIDE-BY-SIDE
    # ════════════════════════
    st.markdown("---")
    st.markdown("### 🗂️ Kelompok per Segmen (Side-by-Side)")
    st.caption("Tiap TPK diassign ke segmen berdasarkan ranking rata-rata TS: sepertiga bawah=Rendah, tengah=Sedang, atas=Tinggi.")

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
                    .reset_index()
                )
                
                s_html = "<div style='overflow-x:auto;'><table style='width:100%; border-collapse:collapse; background:white; font-size:13px;'>"
                s_html += "<tr style='background:#f1f3f5; border-bottom:2px solid #dee2e6;'>"
                for col in _tbl.columns:
                    s_html += f"<th style='padding:8px; text-align:left; color:#212529;'>{col}</th>"
                s_html += "</tr>"
                
                for _, s_row in _tbl.iterrows():
                    s_html += "<tr style='border-bottom:1px solid #e9ecef;'>"
                    for col in _tbl.columns:
                        val = s_row[col]
                        if col == "Kelompok":
                            s_html += f"<td style='padding:8px; color:#333; font-weight:500;'>{val}</td>"
                        else:
                            bg_c, text_c = get_cell_color(col, val)
                            val_str = f"{val:.3f}" if isinstance(val, (float, np.floating)) else str(val)
                            s_html += f"<td style='padding:8px; background:{bg_c}; color:{text_c}; text-align:center;'>{val_str}</td>"
                    s_html += "</tr>"
                s_html += "</table></div>"
                
                st.markdown(s_html, unsafe_allow_html=True)

    # ════════════════════════
    # BAGIAN 4 — REKOMENDASI (DENGAN PILIHAN VARIABEL/CATATAN)
    # ════════════════════════
    st.markdown("---")
    st.markdown("### 💡 Rekomendasi per Segmen")

    SEG_LIST = [
        ("Kualitas Rendah 🔴", "#fdf2f2", "#721c24", "#c0392b"),
        ("Kualitas Sedang 🟡", "#fffdf0", "#856404", "#f39c12"),
        ("Kualitas Tinggi 🟢", "#f2fdf5", "#155724", "#27ae60"),
    ]

    df_rek = get_rekomendasi_segmen()
    rek_db = {row["segmen"]: row["rekomendasi"] for _, row in df_rek.iterrows()} if not df_rek.empty else {}

    if ROLE == "admin":
        st.caption("Pilih variabel yang perlu diberi catatan, lalu masukkan rekomendasinya untuk tiap segmen.")

        VAR_OPTIONS = ["Kadar Air (KA)", "Kadar Lemak (FAT)", "Solid Non Fat (SNF)", "Total Solid (TS)"]

        for seg_lbl, bg, tx, border_clr in SEG_LIST:
            tpk_list = tpk_chars[tpk_chars["Segmen_TPK"] == seg_lbl]["NAMA_KELOMPOK"].tolist()
            tpk_str  = ", ".join(tpk_list) if tpk_list else "-"

            st.markdown(
                f"<div style='background:{bg};border-left:5px solid {border_clr};"
                f"padding:12px 18px;border-radius:8px;margin-bottom:6px;'>"
                f"<div style='font-weight:700;font-size:15px;color:{tx}'>{seg_lbl}</div>"
                f"<div style='font-size:13px;color:{tx}'>TPK: {tpk_str}</div>"
                f"</div>",
                unsafe_allow_html=True)

            selected_vars = st.multiselect(
                f"Pilih variabel yang perlu dicatat untuk {seg_lbl}",
                options=VAR_OPTIONS,
                key=f"vars_{seg_lbl}"
            )

            _saved = rek_db.get(seg_lbl, "")
            _key   = f"rek_input_{seg_lbl}"
            _input = st.text_area(
                f"Tulis rekomendasi penanganan untuk {seg_lbl}",
                value=_saved,
                height=120,
                key=_key,
                placeholder="Tulis rekomendasi penanganan...")

            _col_btn, _col_info = st.columns([1, 3])
            with _col_btn:
                if st.button("💾 Simpan", key=f"btn_rek_{seg_lbl}", use_container_width=True):
                    final_text = _input.strip()
                    if selected_vars:
                        vars_str = ", ".join(selected_vars)
                        final_text = f"📌 Catatan Variabel ({vars_str}):\n{final_text}"
                    
                    save_rekomendasi_segmen(seg_lbl, final_text, st.session_state["username"])
                    st.success(f"✅ Rekomendasi untuk **{seg_lbl}** berhasil disimpan.")
            with _col_info:
                if seg_lbl in rek_db and rek_db[seg_lbl]:
                    _row = df_rek[df_rek["segmen"] == seg_lbl].iloc[0]
                    st.caption(f"Terakhir diperbarui oleh **{_row['updated_by']}** pada {_row['updated_at']}")
            st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    else:
        st.caption("Rekomendasi penanganan berdasarkan hasil segmentasi.")
        ada_rek = False
        for seg_lbl, bg, tx, border_clr in SEG_LIST:
            tpk_list = tpk_chars[tpk_chars["Segmen_TPK"] == seg_lbl]["NAMA_KELOMPOK"].tolist()
            tpk_str  = ", ".join(tpk_list) if tpk_list else "-"
            rek_text = rek_db.get(seg_lbl, "")

            if not rek_text:
                continue
            ada_rek = True
            st.markdown(
                f"<div style='background:{bg};border-left:5px solid {border_clr};"
                f"padding:14px 18px;border-radius:8px;margin-bottom:14px;'>"
                f"<div style='font-weight:700;font-size:15px;color:{tx};margin-bottom:4px'>{seg_lbl}</div>"
                f"<div style='font-size:13px;color:{tx};margin-bottom:8px'>TPK: {tpk_str}</div>"
                f"<div style='font-size:14px;line-height:1.7;color:#333;white-space:pre-line'>{rek_text}</div>"
                f"</div>",
                unsafe_allow_html=True)
        if not ada_rek:
            st.info("Rekomendasi belum tersedia. Hubungi admin untuk mengisi rekomendasi segmentasi.")
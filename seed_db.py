"""
seed_db.py
==========
Jalankan sekali untuk mengisi database.db dengan data awal:
  - data_fix.xlsx        → tabel master_data
  - Dataset_Master_Susu_Bersih.csv → tabel data_tpk

Cara pakai:
    python seed_db.py

Script ini aman dijalankan berulang — tidak akan duplikat data
kalau database sudah terisi (cek jumlah baris dulu sebelum insert).
"""

import sqlite3
import pandas as pd
import os

DB_PATH       = "database.db"
MASTER_FILE   = "data_coba.xlsx"
TPK_FILE      = "Dataset_Master_Susu_Bersih_coba.csv"

# ── Buat / sambung DB ──
def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db(conn):
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
    conn.commit()


# ════════════════════════════════════════════
# SEED MASTER DATA dari data_fix.xlsx
# ════════════════════════════════════════════
def seed_master(conn):
    n_existing = conn.execute("SELECT COUNT(*) FROM master_data").fetchone()[0]
    if n_existing > 0:
        print(f"⚠️  master_data sudah berisi {n_existing:,} baris — skip seeding.")
        return

    if not os.path.exists(MASTER_FILE):
        print(f"❌ File tidak ditemukan: {MASTER_FILE}")
        return

    print(f"📥 Membaca {MASTER_FILE} ...")
    df = pd.read_excel(MASTER_FILE)

    # Rename Temp. → Temp agar sesuai skema DB
    df = df.rename(columns={"Temp.": "Temp"})

    # Parse tanggal & bulan
    df["Tgl"]   = pd.to_datetime(df["Tgl"], errors="coerce")
    df["Bulan"] = df["Tgl"].dt.to_period("M").astype(str)
    df["Tgl"]   = df["Tgl"].dt.strftime("%Y-%m-%d")

    # Kolom yang ada di DB
    db_cols = ["Tgl","Nama","NoPol","Netto","Segel","Temp","Appearance","TDO",
               "PH","AT","BTB","CT","Antibiotik","TS","SNF","FAT","TPC",
               "Density","Durasi_Menit","Bulan"]

    for col in db_cols:
        if col not in df.columns:
            df[col] = None

    df = df[db_cols].dropna(subset=["Tgl"])
    df.to_sql("master_data", conn, if_exists="append", index=False)
    conn.commit()
    print(f"✅ {len(df):,} baris berhasil dimasukkan ke master_data.")


# ════════════════════════════════════════════
# SEED DATA TPK dari Dataset_Master_Susu_Bersih.csv
# ════════════════════════════════════════════
def seed_tpk(conn):
    n_existing = conn.execute("SELECT COUNT(*) FROM data_tpk").fetchone()[0]
    if n_existing > 0:
        print(f"⚠️  data_tpk sudah berisi {n_existing:,} baris — skip seeding.")
        return

    if not os.path.exists(TPK_FILE):
        print(f"❌ File tidak ditemukan: {TPK_FILE}")
        return

    print(f"📥 Membaca {TPK_FILE} ...")
    df = pd.read_csv(TPK_FILE, low_memory=False)

    # Rename KLP SAMPLE → KLP_SAMPLE (no space, for SQLite)
    if "KLP SAMPLE" in df.columns:
        df = df.rename(columns={"KLP SAMPLE": "KLP_SAMPLE"})

    # Pastikan TANGGAL_ASLI ada
    if "TANGGAL_ASLI" not in df.columns and "TANGGAL" in df.columns:
        df["TANGGAL_ASLI"] = pd.to_datetime(df["TANGGAL"], format="%d.%m.%Y", errors="coerce")
        df["TANGGAL_ASLI"] = df["TANGGAL_ASLI"].dt.strftime("%Y-%m-%d")
    elif "TANGGAL_ASLI" in df.columns:
        df["TANGGAL_ASLI"] = pd.to_datetime(df["TANGGAL_ASLI"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "TAHUN" not in df.columns and "TANGGAL_ASLI" in df.columns:
        df["TAHUN"] = pd.to_datetime(df["TANGGAL_ASLI"], errors="coerce").dt.year.astype("Int64")

    # Konversi numerik
    for col in ["KA","FAT","SNF","TS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    db_cols = ["NO","KLP_SAMPLE","KA","FAT","SNF","TS",
               "NAMA_KELOMPOK","TANGGAL","WAKTU","TANGGAL_ASLI","TAHUN"]

    for col in db_cols:
        if col not in df.columns:
            df[col] = None

    df = df[db_cols].dropna(subset=["TANGGAL_ASLI","KA","FAT","SNF","TS"])
    df.to_sql("data_tpk", conn, if_exists="append", index=False)
    conn.commit()
    print(f"✅ {len(df):,} baris berhasil dimasukkan ke data_tpk.")


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n🗄️  Menginisialisasi database: {DB_PATH}")
    conn = get_conn()
    init_db(conn)

    print("\n── MASTER DATA ──────────────────────")
    seed_master(conn)

    print("\n── DATA TPK ─────────────────────────")
    seed_tpk(conn)

    conn.close()
    print("\n🎉 Seeding selesai!")
    print(f"   File: {os.path.abspath(DB_PATH)}")
    print("   Letakkan database.db di folder yang sama dengan app.py sebelum menjalankan Streamlit.\n")

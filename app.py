import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import traceback
import sqlite3
from datetime import datetime

st.set_page_config(page_title="Penilaian Model ML Mahasiswa", layout="centered")
st.title("📊 Penilaian Otomatis Model Machine Learning Mahasiswa")

# -------------------------------
# DATABASE SETUP
# -------------------------------
def init_db():
    conn = sqlite3.connect("penilaian.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS penilaian (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nim TEXT,
            accuracy REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def simpan_nilai(nim, accuracy):
    conn = sqlite3.connect("penilaian.db")
    c = conn.cursor()
    c.execute("INSERT INTO penilaian (nim, accuracy) VALUES (?, ?)", (nim, accuracy))
    conn.commit()
    conn.close()

# Inisialisasi database
init_db()

# -------------------------------
# LOAD TEST DATA (FROM DOSEN)
# -------------------------------
@st.cache_data
def load_test_data():
    try:
        df = pd.read_csv("test.csv")  # File test dari dosen
        X = df.drop("label", axis=1)
        y = df["label"]
        return X, y
    except Exception as e:
        st.error("❌ Gagal memuat data test: pastikan file `test.csv` tersedia.")
        return None, None

X_test, y_test = load_test_data()

# -------------------------------
# FORM INPUT MAHASISWA
# -------------------------------
st.subheader("🧑‍🎓 Input Mahasiswa")
nim = st.text_input("Masukkan NIM")

uploaded_model = st.file_uploader("📤 Upload model.pkl (Pipeline scikit-learn)", type=["pkl"])

if nim and uploaded_model and X_test is not None:
    try:
        # Load pipeline
        model = joblib.load(uploaded_model)

        # Cek metode predict
        if not hasattr(model, "predict"):
            raise ValueError("Model tidak memiliki metode predict(). Pastikan ini pipeline scikit-learn.")

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Akurasi model: {acc:.2%}")

        # Tampilkan hasil prediksi
        st.subheader("🔍 Contoh Hasil Prediksi")
        results = X_test.copy()
        results["Label Asli"] = y_test.values
        results["Prediksi"] = y_pred
        st.dataframe(results.head(10))

        # Simpan ke database
        simpan_nilai(nim, acc)
        st.info("📝 Nilai berhasil disimpan ke database.")

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat memproses model.")
        st.code(traceback.format_exc())

# -------------------------------
# OPSIONAL: Tampilkan Hasil Semua Mahasiswa
# -------------------------------
st.subheader("📋 Rekap Nilai Mahasiswa (sementara)")
if st.checkbox("Tampilkan semua nilai"):
    conn = sqlite3.connect("penilaian.db")
    df_nilai = pd.read_sql_query("SELECT * FROM penilaian ORDER BY timestamp DESC", conn)
    conn.close()
    st.dataframe(df_nilai)

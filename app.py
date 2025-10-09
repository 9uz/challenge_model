import streamlit as st
import pandas as pd
import joblib
import traceback
import time
import hashlib
import hmac
import mysql.connector
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# KONFIGURASI APLIKASI
# -------------------------------
st.set_page_config(page_title="Penilaian Model ML Mahasiswa", layout="centered")
st.title("📊 Penilaian Otomatis Model Machine Learning Mahasiswa")

# -------------------------------
# KONEKSI DATABASE MENGGUNAKAN st.secrets
# -------------------------------
def get_connection():
    config = st.secrets["mysql"]
    return mysql.connector.connect(
        host=config["host"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        port=config["port"],
    )

# -------------------------------
# INIT DATABASE
# -------------------------------
def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS penilaian (
            id INT AUTO_INCREMENT PRIMARY KEY,
            nim VARCHAR(20),
            inisial VARCHAR(50),
            accuracy FLOAT,
            model_hash VARCHAR(64) UNIQUE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def simpan_nilai(nim, inisial, accuracy, model_hash):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO penilaian (nim, inisial, accuracy, model_hash) VALUES (%s, %s, %s, %s)", (nim, inisial, accuracy, model_hash))
    conn.commit()
    conn.close()

def model_sudah_ada(model_hash):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM penilaian WHERE model_hash = %s", (model_hash,))
    result = c.fetchone()
    conn.close()
    return result[0] > 0

init_db()

# -------------------------------
# LOAD TEST DATA
# -------------------------------
@st.cache_data
def load_test_data():
    try:
        df = pd.read_csv("test.csv")
        X = df.drop("label", axis=1)
        y = df["label"]
        return X, y
    except Exception as e:
        st.error("❌ Gagal memuat data test: pastikan file `test.csv` tersedia.")
        return None, None

X_test, y_test = load_test_data()

# -------------------------------
# Fungsi hash model
# -------------------------------
def hash_model(file_buffer):
    file_buffer.seek(0)
    bytes_data = file_buffer.read()
    file_buffer.seek(0)
    return hashlib.sha256(bytes_data).hexdigest()

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["🧑‍🎓 Mahasiswa", "📋 Rekap Scoreboard Mahasiswa"])

with tab1:
    st.subheader("🧑‍🎓 Input Mahasiswa")

    nim = st.text_input("Masukkan NIM (9 digit angka)")
    inisial = st.text_input("Masukkan Inisial (misalnya: R. Hadi)")
    uploaded_model = st.file_uploader("📤 Upload model.pkl (Pipeline scikit-learn)", type=["pkl"])

    valid_nim = nim.isdigit() and len(nim) == 9
    valid_inisial = len(inisial.strip()) > 0
    model_uploaded = uploaded_model is not None

    if nim and not valid_nim:
        st.warning("⚠️ Format NIM harus terdiri dari 10 digit angka.")

    if valid_nim and valid_inisial and model_uploaded:
        submit = st.button("✅ Submit Model")

        if submit:
            if X_test is None:
                st.error("❌ Data test tidak tersedia.")
            else:
                try:
                    with st.spinner("⏳ Memproses model..."):
                        progress_bar = st.progress(0)
                        for percent_complete in range(0, 60, 10):
                            time.sleep(0.1)
                            progress_bar.progress(percent_complete)

                        # Hash model
                        model_hash = hash_model(uploaded_model)

                        if model_sudah_ada(model_hash):
                            st.error("❌ Model ini sudah pernah diupload sebelumnya. Upload model lain.")
                        else:
                            model = joblib.load(uploaded_model)
                            if not hasattr(model, "predict"):
                                raise ValueError("Model tidak memiliki metode predict(). Pastikan ini pipeline scikit-learn.")

                            progress_bar.progress(70)
                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            progress_bar.progress(85)

                            simpan_nilai(nim, inisial, acc, model_hash)

                            progress_bar.progress(100)

                            st.success(f"✅ Akurasi model: {acc:.2%}")
                            st.toast("📝 Nilai berhasil disimpan ke database.")

                            st.subheader("🔍 Contoh Hasil Prediksi")
                            results = X_test.copy()
                            results["Label Asli"] = y_test.values
                            results["Prediksi"] = y_pred
                            st.dataframe(results.head(5))

                            st.subheader("📄 Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=False)
                            st.text(report)

                            st.subheader("🧮 Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            st.pyplot(fig)

                except Exception as e:
                    st.error("❌ Terjadi kesalahan saat memproses model.")
                    st.code(traceback.format_exc())
    else:
        st.info("ℹ️ Masukkan NIM, inisial, dan upload model untuk submit.")

    # Histori board mahasiswa
    if nim and valid_nim:
        conn = get_connection()
        df_histori = pd.read_sql(f"SELECT accuracy, timestamp FROM penilaian WHERE nim = %s ORDER BY timestamp DESC", conn, params=(nim,))
        conn.close()

        if not df_histori.empty:
            st.subheader(f"📜 Histori Nilai untuk NIM {nim}")
            df_histori["Ranking"] = df_histori["accuracy"].rank(method='dense', ascending=False).astype(int)
            cols = ["Ranking"] + [col for col in df_histori.columns if col != "Ranking"]
            st.dataframe(df_histori[cols])

with tab2:
    st.subheader("📋 Rekap Scoreboard Mahasiswa")

    conn = get_connection()
    df_nilai = pd.read_sql("SELECT nim, accuracy, timestamp FROM penilaian ORDER BY accuracy DESC, timestamp ASC", conn)
    conn.close()

    if df_nilai.empty:
        st.info("Belum ada Scoreboard yang tersimpan.")
    else:
        df_nilai["Ranking"] = range(1, len(df_nilai) + 1)
        cols = ["Ranking"] + [col for col in df_nilai.columns if col != "Ranking"]
        st.dataframe(df_nilai[cols])

        password_hash = st.secrets.get("DOWNLOAD_PASSWORD_HASH")

        with st.expander("🔒 Download Rekap Data (khusus dosen)"):
            if not password_hash:
                st.error("⚠️ Password untuk download belum dikonfigurasi. Hubungi admin.")
            else:
                input_pw = st.text_input("Masukkan password untuk mengunduh data:", type="password")
                if input_pw:
                    input_hash = hashlib.sha256(input_pw.encode()).hexdigest()
                    if hmac.compare_digest(input_hash, password_hash):
                        csv = df_nilai[cols].to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Download Rekap CSV", data=csv,
                                           file_name="rekap_nilai_mahasiswa.csv", mime="text/csv")
                    else:
                        st.error("❌ Password salah.")

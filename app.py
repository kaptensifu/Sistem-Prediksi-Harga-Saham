import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import joblib
from transformers import pipeline
from pandas.tseries.offsets import BDay 
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# 1. DEFINISI CUSTOM LAYER UNTUK VAE
# ============================================================
@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """Reparameterization trick: z = μ + σ * ε"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ============================================================
# 2. CACHING & DATA MANAGEMENT
# ============================================================
@st.cache_resource
def load_models():
    scaler_X = joblib.load('minmax_scaler_x.pkl')
    scaler_y = joblib.load('minmax_scaler_y.pkl') 
    
    encoder = keras.models.load_model('encoder_4.keras', custom_objects={'Sampling': Sampling})
    bilstm = keras.models.load_model('model_vae_bilstm_f4.keras')

    lokasi_kode_saat_ini = os.getcwd()
    path_model_lokal = os.path.join(lokasi_kode_saat_ini, "model_finbert")

    finbert = pipeline(
        "sentiment-analysis", 
        model=path_model_lokal, 
        tokenizer=path_model_lokal, 
        top_k=None,
        truncation=True,
        max_length=512
    )
        
    return scaler_X, scaler_y, encoder, bilstm, finbert

@st.cache_data
def load_historical_data():
    df = pd.read_csv('GOTO_Historical_Data_web.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def save_to_csv(date, o, h, l, c, v, sentiment):
    file_path = 'GOTO_Historical_Data_web.csv'
    new_row = pd.DataFrame({
        'Date': [pd.to_datetime(date).strftime('%Y-%m-%d')],
        'Open': [o], 'High': [h], 'Low': [l], 'Close': [c], 'Volume': [v],
        'Nilai_Sentimen': [sentiment]
    })
    new_row.to_csv(file_path, mode='a', header=False, index=False)
    load_historical_data.clear() 

def undo_last_entry():
    file_path = 'GOTO_Historical_Data_web.csv'
    df = pd.read_csv(file_path)
    if len(df) > 30: # Sisakan minimal 30 data agar grafik perbandingan 15 hari aman
        df = df.iloc[:-1] 
        df.to_csv(file_path, index=False)
        load_historical_data.clear()
        return True
    return False

# ============================================================
# 3. FUNGSI PREPROCESSING & PREDIKSI BATCH (16 WINDOW)
# ============================================================
def get_sentiment_score(text, finbert_model):
    if not text.strip(): 
        return 0.0
    results = finbert_model(text)[0]
    scores = {res['label']: res['score'] for res in results}
    sentiment_val = scores.get('positive', 0.0) - scores.get('negative', 0.0)
    return sentiment_val

def preprocess_and_predict(input_date, o, h, l, c, v, sentiment_score, full_hist_df, models):
    scaler_X, scaler_y, encoder, bilstm = models[:4]
    
    # Ambil 30 data historis untuk keperluan grafik
    hist_df = full_hist_df.tail(30).reset_index(drop=True)
    if len(hist_df) < 30:
        return None, None, "Data historis CSV kurang dari 30 hari."

    new_data = pd.DataFrame({
        'Date': [pd.to_datetime(input_date)],
        'Open': [o], 'High': [h], 'Low': [l], 'Close': [c], 'Volume': [v],
        'Nilai_Sentimen': [sentiment_score]
    })

    combined_df_raw = pd.concat([hist_df, new_data], ignore_index=True)
    raw_close_prices = combined_df_raw['Close'].values
    raw_dates = combined_df_raw['Date'].values

    # Hitung Log Return
    combined_df = combined_df_raw.copy()
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    combined_df[num_cols] = np.log(combined_df[num_cols] / combined_df[num_cols].shift(1))
    combined_df = combined_df.dropna().reset_index(drop=True)  # 30 baris

    # Normalisasi & encode SEMUA 30 baris (untuk grafik 15 prediksi historis)
    ohlcv_features = combined_df[num_cols].values
    scaled_ohlcv = scaler_X.transform(ohlcv_features)
    sentiment_feature = combined_df[['Nilai_Sentimen']].values
    scaled_features = np.hstack((scaled_ohlcv, sentiment_feature))
    z_mean_full, _, _ = encoder.predict(scaled_features, verbose=0)  # (30, LATENT_DIM)

    # ============================================================
    # KUNCI PERBAIKAN:
    # Encode ulang window terakhir menggunakan konteks tail(16)
    # agar sentimen hari ini punya bobot seperti kode lama
    # ============================================================
    hist_df_16 = full_hist_df.tail(15).reset_index(drop=True)  # ambil 15 historis
    combined_16_raw = pd.concat([hist_df_16, new_data], ignore_index=True)  # 16 baris
    
    combined_16 = combined_16_raw.copy()
    combined_16[num_cols] = np.log(combined_16[num_cols] / combined_16[num_cols].shift(1))
    combined_16 = combined_16.dropna().reset_index(drop=True)  # 15 baris
    
    ohlcv_16 = combined_16[num_cols].values
    scaled_ohlcv_16 = scaler_X.transform(ohlcv_16)
    sentiment_16 = combined_16[['Nilai_Sentimen']].values
    scaled_features_16 = np.hstack((scaled_ohlcv_16, sentiment_16))
    z_mean_16, _, _ = encoder.predict(scaled_features_16, verbose=0)  # (15, LATENT_DIM)

    # Bentuk 16 windows:
    # - Window 0–14  : dari z_mean_full (untuk grafik historis)
    # - Window 15    : dari z_mean_16   (untuk prediksi besok, sentimen hari ini dominan)
    timestep = 15
    windows = []
    for i in range(15):  # 15 window pertama dari konteks 30 hari
        windows.append(z_mean_full[i : i + timestep])
    windows.append(z_mean_16[0:15])  # window ke-16 dari konteks 16 hari (seperti kode lama)

    window_data_batch = np.array(windows)  # (16, 15, LATENT_DIM)

    # BiLSTM prediksi
    pred_scaled_batch = bilstm.predict(window_data_batch, verbose=0)
    pred_log_return_batch = scaler_y.inverse_transform(pred_scaled_batch).flatten()

    # Inverse ke harga asli
    base_prices = raw_close_prices[15:31]
    pred_prices = base_prices * np.exp(pred_log_return_batch)

    # Siapkan plot data
    future_date = pd.to_datetime(input_date) + BDay(1)
    plot_dates = list(pd.to_datetime(raw_dates[16:])) + [future_date]
    actual_plot_prices = list(raw_close_prices[16:]) + [np.nan]

    plot_data = {
        'dates': plot_dates,
        'actual': actual_plot_prices,
        'predicted': pred_prices
    }

    final_pred_price = pred_prices[-1]
    if sentiment_score > 0.7:
        final_pred_price += 1.23
    elif 0.3 < sentiment_score <= 0.7:
        final_pred_price += 0.74
    elif sentiment_score < -0.7:
        final_pred_price -= 1.11
    elif -0.7 <= sentiment_score < -0.3:
        final_pred_price -= 0.57
      
    final_pred_price = round(final_pred_price)

    plot_data['predicted'][-1] = final_pred_price

    final_pred_price = pred_prices[-1]    

    final_window_data = window_data_batch[-1:]

    return final_pred_price, final_window_data, plot_data

# ============================================================
# 4. TAMPILAN WEB STREAMLIT 
# ============================================================
st.set_page_config(page_title="Prediksi Saham VAE-BiLSTM", page_icon="📈", layout="wide")

st.title("📈 Sistem Prediksi Harga Saham Volatil")
st.markdown("""
Aplikasi web ini menggunakan arsitektur **VAE-BiLSTM** untuk memprediksi harga saham, 
dikombinasikan dengan ekstraksi sentimen berita menggunakan **FinBERT**.
""")
st.divider() 

# Load resource
with st.spinner("Memuat model dan data historis..."):
    models = load_models()
    full_hist_df = load_historical_data()

# Validasi Tanggal Otomatis
last_date_in_csv = full_hist_df['Date'].iloc[-1]
expected_next_date = last_date_in_csv + BDay(1)

# Sidebar
st.sidebar.subheader("🛠️ Manajemen Data Historis")
st.sidebar.info(f"Tanggal terakhir di CSV:\n**{last_date_in_csv.strftime('%d %B %Y')}**")

if st.sidebar.button("⚠️ Hapus Data Terakhir", use_container_width=True):
    if undo_last_entry():
        st.sidebar.success("Baris data terakhir berhasil dihapus dari CSV!")
        st.rerun()
    else:
        st.sidebar.error("Gagal menghapus. Harus tersisa minimal 30 data di CSV.")

# Main UI
st.subheader("Data Hari Ini")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("##### 📅 Tanggal & Data Saham (OHLCV)")
    with st.container(border=True): 
        input_date = st.date_input("Tanggal Input", value=expected_next_date.date())
        
        c_open, c_high = st.columns(2)
        input_open = c_open.number_input("Open (Lot)", value=0)
        input_high = c_high.number_input("High (Lot)", value=0)
        
        c_low, c_close = st.columns(2)
        input_low = c_low.number_input("Low (Lot)", value=0)
        input_close = c_close.number_input("Close (Lot)", value=0)
        
        input_volume = st.number_input("Volume", value=0)

with col2:
    st.markdown("##### 📰 Sentimen Pasar")
    with st.container(border=True):
        st.info("Masukkan teks judul berita terkait saham GOTO hari ini.")
        input_news_id = st.text_area("Teks Berita", height=200)

st.markdown("<br>", unsafe_allow_html=True) 

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("💾 Simpan Data & Prediksi Harga", use_container_width=True, type="primary")

# Eksekusi Prediksi
if predict_button:
    # --- Validasi Input ---
    validation_errors = []

    if input_open <= 0:
        validation_errors.append("**Open** belum diisi atau bernilai 0.")
    if input_high <= 0:
        validation_errors.append("**High** belum diisi atau bernilai 0.")
    if input_low <= 0:
        validation_errors.append("**Low** belum diisi atau bernilai 0.")
    if input_close <= 0:
        validation_errors.append("**Close** belum diisi atau bernilai 0.")
    if input_volume <= 0:
        validation_errors.append("**Volume** belum diisi atau bernilai 0.")
    if not input_news_id.strip():
        validation_errors.append("**Teks Berita** belum diisi.")

    if validation_errors:
        st.error("❌ **Validasi Gagal:** Semua field wajib diisi sebelum prediksi dapat dijalankan.")
        for err in validation_errors:
            st.warning(f"• {err}")
    elif pd.to_datetime(input_date).date() != expected_next_date.date():
        st.error(f"❌ **Validasi Gagal:** Harap masukkan data untuk tanggal **{expected_next_date.strftime('%d %B %Y')}** agar urutan *Time-Series* tidak rusak.")
    else:
        with st.spinner('Menerjemahkan teks, memproses prediksi batch, dan menggambar grafik...'):
            translated_news = ""
            sentiment_score = 0.0
            
            if input_news_id.strip():
                try:
                    translator = GoogleTranslator(source='id', target='en')
                    translated_news = translator.translate(input_news_id)
                    st.info(f"**Teks Terjemahan :** {translated_news}")
                    
                    finbert_tokenizer = models[4].tokenizer
                    tokens = finbert_tokenizer.tokenize(translated_news)
                    if len(tokens) > 512:
                        st.warning(f"⚠️ **Peringatan Batas Token:** Teks memiliki **{len(tokens)} token**. Dipotong otomatis.")
                except Exception as e:
                    st.error(f"Kesalahan translasi: {e}")
                    translated_news = input_news_id 
                
                sentiment_score = get_sentiment_score(translated_news, models[4])
                
            # Prediksi Utama
            pred_price, window_data, plot_data = preprocess_and_predict(
                input_date, input_open, input_high, input_low, input_close, input_volume, 
                sentiment_score, full_hist_df, models
            )
            
            if pred_price is None:
                st.error(plot_data) # plot_data berisi pesan error jika data < 30
            else:
                save_to_csv(input_date, input_open, input_high, input_low, input_close, input_volume, sentiment_score)
                
                next_bday = pd.to_datetime(input_date) + BDay(1)
                next_date_str = next_bday.strftime('%d %B %Y')
                
                st.divider()
                st.success(f"✅ Data tanggal {pd.to_datetime(input_date).strftime('%d %b %Y')} berhasil disimpan!")
                
                # BAGIAN 1: METRIK PREDIKSI
                res_col1, res_col2 = st.columns([2, 1])
                with res_col1:
                    st.markdown(f"### Prediksi Harga *Close* untuk:")
                    st.markdown(f"<h2 style='color: #1f77b4;'>🗓️ {next_date_str}</h2>", unsafe_allow_html=True)
                with res_col2:
                    st.metric(label="Estimasi Harga Close", value=f"{pred_price:,.0f} Lot")
                
                # BAGIAN 2: GRAFIK MATPLOTLIB
                st.markdown("---")
                st.subheader("📊 Perbandingan Harga Aktual vs Prediksi")
                
                fig, ax = plt.subplots(figsize=(12, 5))
                
                # Plot data
                ax.plot(plot_data['dates'], plot_data['actual'], marker='o', label='Harga Aktual', color='#1f77b4', linewidth=2)
                ax.plot(plot_data['dates'], plot_data['predicted'], marker='X', linestyle='--', label='Harga Prediksi Model', color='#ff7f0e', linewidth=2)
                
                # Format sumbu X agar rapi
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
                plt.xticks(plot_data['dates'], rotation=45)
                
                # Kustomisasi Grid dan Label
                ax.set_xlabel("Tanggal Transaksi")
                ax.set_ylabel("Harga Close (Lot)")
                ax.legend(loc='upper left')
                ax.grid(True, linestyle=':', alpha=0.7)
                
                # Lingkari prediksi terakhir (titik paling kanan) untuk menonjolkannya
                ax.scatter(plot_data['dates'][-1], plot_data['predicted'][-1], color='red', s=150, zorder=5, label='Prediksi Hari Esok')
                
                plt.tight_layout()
                st.pyplot(fig)


from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Hindari error GUI di Flask
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import glob
from datetime import datetime

app = Flask(__name__)

# Baca dataset dari file CSV
df = pd.read_csv("dataset_dengan_emosi.csv")
# Konversi kolom 'timestamp' menjadi format tanggal dan waktu
df['timestamp'] = pd.to_datetime(df['timestamp'])
# Tambahkan kolom 'date' yang berisi tanggal saja dari 'timestamp'
df['date'] = df['timestamp'].dt.date
# Tambahkan kolom 'hour' yang berisi jam saja dari 'timestamp'
df['hour'] = df['timestamp'].dt.hour

# Siapkan folder untuk menyimpan file uploads
UPLOAD_FOLDER = "static/uploads"
# Pastikan folder uploads ada, jika tidak ada maka buat folder baru
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi analyzer VADER untuk analisis sentimen
analyzer = SentimentIntensityAnalyzer()

# Daftar kata kunci positif (konteks kasus Willie Salim)
positive_keywords = [
    'niat baik', 'tulus', 'berbagi', 'berderma', 'peduli', 'mulia',
    'hebat', 'mantap', 'inspiratif', 'terima kasih', 'bagus', 'baik hati',
    'respect', 'apresiasi', 'support', 'dukung', 'semangat', 'jangan menyerah',
    'tabah', 'sabar', 'pemaaf', 'ikhlas', 'niat mulia', 'niat berbagi',
    'konten positif', 'edukatif', 'bermanfaat', 'semoga sukses',
    'tetap berkarya', 'jangan kapok', 'berkah', 'terharu', 'bangga',
    'penuh cinta', 'peaceful', 'pengertian', 'bijak', 'adil', 'gentle',
    'tenang', 'tidak menyalahkan', 'klarifikasi baik', 'niat bagus', 'positif thinking',
    'pemaafan', 'rendah hati', 'simpati', 'empati', 'damai', 'terimakasih', 'terimakasi', 'maaf'
]

@app.route("/")
def index():
    # Tampilkan halaman index
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    # Dapatkan komentar dari form upload
    komentar = request.form["komentar"]
    # Konversi komentar menjadi huruf kecil untuk memudahkan pencarian kata kunci
    lower_komentar = komentar.lower()

    # Cek apakah ada kata positif dalam komentar
    ditemukan_kata_positif = [kw for kw in positive_keywords if kw in lower_komentar]

    # Jika ditemukan kata positif, maka hasilnya adalah "Positif", jika tidak maka "Negatif"
    if ditemukan_kata_positif:
        hasil = "Positif"
    else:
        hasil = "Negatif"

    # Tampilkan hasil upload dengan detail komentar, hasil, dan kata positif yang ditemukan
    return render_template(
        "hasil_upload.html",
        komentar=komentar,
        hasil=hasil,
        kata_positif=ditemukan_kata_positif
    )


@app.route("/visualisasi")
def visualisasi():
    # Bersihkan file lama di folder uploads
    for file in glob.glob(os.path.join(UPLOAD_FOLDER, "*.png")):
        os.remove(file)

    plots = []

    def save_plot(fig, name):
        # Simpan plot sebagai file PNG dengan nama yang unik berdasarkan waktu
        filename = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        path = os.path.join(UPLOAD_FOLDER, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        # Kembalikan path relatif dari file yang disimpan
        return f"uploads/{filename}"

    # 1. Distribusi Sentimen
    fig = plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label', hue='label', palette='Set2', legend=False)
    plt.title('Distribusi Sentimen')
    plt.xlabel('Label Sentimen')
    plt.ylabel('Jumlah Komentar')
    plt.xticks(ticks=[0,1,2], labels=['Negatif', 'Netral', 'Positif'])
    plots.append(save_plot(fig, 'sentimen'))

    # 2. Aktivitas Komentar per Hari
    fig = plt.figure(figsize=(12, 4))
    df_per_day = df.groupby('date').size()
    df_per_day.plot()
    plt.title('Aktivitas Komentar per Hari')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Komentar')
    plots.append(save_plot(fig, 'per_hari'))

    # 3. Aktivitas Komentar per Jam
    fig = plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='hour', hue='hour', palette='viridis', legend=False)
    plt.title('Distribusi Komentar per Jam')
    plt.xlabel('Jam (0-23)')
    plt.ylabel('Jumlah Komentar')
    plots.append(save_plot(fig, 'per_jam'))

    # 4. Distribusi Emosi
    fig = plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='emosi', hue='emosi', palette='Set2', legend=False)
    plt.title("Distribusi Emosi dalam Komentar")
    plt.xlabel("Emosi")
    plt.ylabel("Jumlah Komentar")
    plots.append(save_plot(fig, 'emosi'))

    # 5. WordCloud untuk setiap label
    label_map = {
        0: "Negatif",
        1: "Netral",
        2: "Positif"
    }

    for label in df['label'].unique():
        label_text = label_map.get(label, str(label))
        text = ' '.join(df[df['label'] == label]['komentar'].dropna().astype(str))
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'WordCloud Sentimen: {label_text}', fontsize=16)
            plots.append(save_plot(fig, f'wordcloud_{label_text.lower()}'))

    # 6. Distribusi per Platform
    if 'platform' in df.columns:
        fig = plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='platform', hue='platform', palette='pastel', legend=False)
        plt.title('Distribusi Komentar per Platform')
        plt.xlabel('Platform')
        plt.ylabel('Jumlah Komentar')
        plots.append(save_plot(fig, 'platform'))

    # Tampilkan halaman visualisasi dengan semua plot yang telah disimpan
    return render_template("visualisasi.html", plots=plots)

if __name__ == "__main__":
    app.run(debug=True)

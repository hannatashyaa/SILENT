# Gunakan base image Python 3.12 secara eksplisit
FROM python:3.12-slim-buster

# Instal dependensi sistem yang umum dibutuhkan oleh pustaka ML
# Pastikan ini berada di awal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libv4l-dev \
    libgtk2.0-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Bagian ini adalah modifikasi utama untuk masalah distutils ----
# Ini adalah upaya terakhir untuk memastikan setuptools selalu mutakhir.
# Walaupun kita akan pakai venv, ada kalanya build backend masih memanggil
# setuptools dari base environment atau temporary environment.
RUN pip install --no-input --upgrade setuptools pip wheel

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Atur variabel lingkungan PATH
ENV NIXPACKS_PATH=/opt/venv/bin:$NIXPACKS_PATH

# Buat virtual environment
RUN python -m venv --copies /opt/venv

# AKTIFKAN VIRTUAL ENVIRONMENT DAN LAKUKAN UPGRADE LAGI DI DALAMNYA
# Ini adalah baris penting yang sudah kita coba sebelumnya, pastikan ada.
RUN . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel

# Salin file requirements.txt
COPY requirements.txt .

# Sekarang, install dependensi dari requirements.txt.
# Harapannya, pip/setuptools sudah sangat up-to-date di sini.
RUN . /opt/venv/bin/activate && \
    pip install -r requirements.txt

# Salin sisa kode aplikasi setelah semua dependensi terinstal
COPY . .

# Atur command untuk menjalankan aplikasi Anda
CMD ["python", "app.py"]

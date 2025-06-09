# OPSI 1: Coba yang paling umum untuk slim
FROM python:3.12-slim

# OPSI 2: Jika OPSI 1 gagal, coba ini (menggunakan Debian Bookworm)
# FROM python:3.12-bookworm

# OPSI 3: Jika OPSI 1 & 2 gagal, coba ini (slim di atas Bookworm)
# FROM python:3.12-slim-bookworm

# ... sisa Dockerfile Anda tetap sama persis seperti yang terakhir saya berikan
# (termasuk apt-get install dan dua langkah pip install)

# Instal dependensi sistem yang umum dibutuhkan oleh pustaka ML
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
RUN pip install --no-input --upgrade setuptools pip wheel

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Atur variabel lingkungan PATH
ENV NIXPACKS_PATH=/opt/venv/bin:$NIXPACKS_PATH

# Buat virtual environment
RUN python -m venv --copies /opt/venv

# AKTIFKAN VIRTUAL ENVIRONMENT DAN LAKUKAN UPGRADE LAGI DI DALAMNYA
RUN . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel

# Salin file requirements.txt
COPY requirements.txt .

# Sekarang, install dependensi dari requirements.txt.
RUN . /opt/venv/bin/activate && \
    pip install -r requirements.txt

# Salin sisa kode aplikasi setelah semua dependensi terinstal
COPY . .

# Atur command untuk menjalankan aplikasi Anda
CMD ["python", "app.py"]

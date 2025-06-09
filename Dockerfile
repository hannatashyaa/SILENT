# Gunakan base image Python 3.12 secara eksplisit
FROM python:3.12-slim-buster

# Instal dependensi sistem yang umum dibutuhkan oleh pustaka ML (seperti OpenCV, TensorFlow)
# Ini harus selalu dilakukan di awal, sebelum instalasi paket Python yang memerlukan build tools.
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
    # Tambahan penting untuk memastikan kompatibilitas setuptools yang lebih baik
    git \
    # Clean up apt cache
    && rm -rf /var/lib/apt/lists/*

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Atur variabel lingkungan PATH
ENV NIXPACKS_PATH=/opt/venv/bin:$NIXPACKS_PATH

# Buat virtual environment terlebih dahulu
RUN python -m venv --copies /opt/venv

# AKTIFKAN VIRTUAL ENVIRONMENT DAN LAKUKAN UPGRADE AWAL SETELAH ITU
# Ini adalah langkah KRUSIAL untuk masalah distutils di 3.12.
# Pastikan pip, setuptools, dan wheel benar-benar mutakhir di dalam venv sebelum apapun.
RUN . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel

# Salin file requirements.txt setelah venv dan pip/setuptools diupgrade
COPY requirements.txt .

# Sekarang, install dependensi dari requirements.txt.
# Karena pip/setuptools sudah up-to-date, seharusnya mereka bisa menangani distutils.
RUN . /opt/venv/bin/activate && \
    pip install -r requirements.txt

# Salin sisa kode aplikasi setelah semua dependensi terinstal
COPY . .

# Atur command untuk menjalankan aplikasi Anda (contoh untuk Backend Flask)
CMD ["python", "app.py"]

# Gunakan base image Python 3.12 secara eksplisit
FROM python:3.12-slim-buster

# Instal dependensi sistem yang umum dibutuhkan oleh pustaka ML (seperti OpenCV, TensorFlow)
# Termasuk build-essential untuk compiler dan library-library lain.
# Perintah ini mungkin perlu disesuaikan tergantung distro Linux dasar (misal Alpine vs Debian/Ubuntu)
# python:3.12-slim-buster menggunakan Debian/Ubuntu base.
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
    && rm -rf /var/lib/apt/lists/*

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Atur variabel lingkungan PATH
ENV NIXPACKS_PATH=/opt/venv/bin:$NIXPACKS_PATH

# Salin file requirements.txt terlebih dahulu untuk memanfaatkan Docker cache layer
COPY requirements.txt .

# Buat virtual environment, aktifkan, dan install semua dependensi Python
# Tetap sertakan pip install --upgrade pip setuptools wheel sebagai safeguard
RUN python -m venv --copies /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Salin sisa kode aplikasi setelah dependensi terinstal
COPY . .

# Atur command untuk menjalankan aplikasi Anda (contoh untuk Backend Flask)
CMD ["python", "app.py"]

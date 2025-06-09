# Gunakan base image Python 3.12 secara eksplisit
FROM python:3.12-slim-buster

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Atur variabel lingkungan PATH
ENV NIXPACKS_PATH=/opt/venv/bin:$NIXPACKS_PATH

# Salin file requirements.txt terlebih dahulu untuk memanfaatkan Docker cache layer
COPY requirements.txt .

# Buat virtual environment, aktifkan, dan install semua dependensi
# TETAPKAN BARIS UPGRADE INI UNTUK DEBUGGING/KOMPATIBILITAS AWAL
RUN python -m venv --copies /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Salin sisa kode aplikasi setelah dependensi terinstal
COPY . .

# Atur command untuk menjalankan aplikasi Anda (contoh untuk Backend Flask)
# Sesuaikan dengan entry point utama Anda, misalnya:
CMD ["python", "app.py"]

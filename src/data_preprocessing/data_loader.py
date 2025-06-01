# silent-ml/src/data_preprocessing/data_loader.py

import os
import pandas as pd
from src.utils.mediapipe_utils import extract_hand_landmarks # Import fungsi dari modul utilitas

def load_and_process_dataset(dataset_root_path, sign_language_type):
    """
    Memuat gambar dari struktur folder dataset, mengekstrak landmark,
    dan mengembalikan DataFrame pandas.

    Args:
        dataset_root_path (str): Path ke folder root dataset (misal: 'data/raw/bisindo').
        sign_language_type (str): Tipe bahasa isyarat ('BISINDO' atau 'SIBI').

    Returns:
        pd.DataFrame: DataFrame yang berisi landmark dan label.
    """
    data = []
    labels = []

    print(f"\nDEBUG: Starting load_and_process_dataset for {sign_language_type}")
    print(f"DEBUG: dataset_root_path received: {dataset_root_path}")

    # Cek apakah dataset_root_path benar-benar ada dan merupakan direktori
    if not os.path.exists(dataset_root_path):
        print(f"ERROR: dataset_root_path '{dataset_root_path}' does not exist.")
        return pd.DataFrame() # Mengembalikan DataFrame kosong jika jalur tidak ada
    if not os.path.isdir(dataset_root_path):
        print(f"ERROR: dataset_root_path '{dataset_root_path}' is not a directory.")
        return pd.DataFrame() # Mengembalikan DataFrame kosong jika bukan direktori


    # Iterasi melalui subfolder (label/kelas)
    # Ini mengasumsikan bahwa di dalam dataset_root_path, ada folder untuk setiap label (misal: A, B, C)
    # dan di dalam folder label tersebut ada gambar.
    try:
        subfolders_or_files_in_root = os.listdir(dataset_root_path)
        print(f"DEBUG: Contents of '{dataset_root_path}': {subfolders_or_files_in_root}")
    except Exception as e:
        print(f"ERROR: Could not list contents of '{dataset_root_path}'. Error: {e}")
        return pd.DataFrame()

    for label_name in sorted(subfolders_or_files_in_root):
        label_path = os.path.join(dataset_root_path, label_name)

        print(f"DEBUG: Checking item: {label_path}")

        if os.path.isdir(label_path):
            print(f"Processing {sign_language_type} - {label_name}...")
            # Iterasi melalui gambar di setiap subfolder
            try:
                images_in_label_folder = os.listdir(label_path)
                print(f"DEBUG: Contents of '{label_path}': {images_in_label_folder}")
                if not images_in_label_folder:
                    print(f"WARNING: Folder '{label_path}' is empty.")
            except Exception as e:
                print(f"ERROR: Could not list contents of '{label_path}'. Error: {e}")
                continue # Lanjutkan ke label berikutnya

            for img_name in images_in_label_folder:
                img_path = os.path.join(label_path, img_name)

                # Pastikan ini adalah file, bukan sub-direktori lain
                if not os.path.isfile(img_path):
                    print(f"WARNING: Skipping non-file item: {img_path}")
                    continue

                # Ekstrak landmark menggunakan fungsi utilitas
                print(f"DEBUG: Attempting to extract landmarks from: {img_path}")
                try:
                    landmarks = extract_hand_landmarks(img_path)

                    if landmarks is not None: # Pastikan landmark berhasil diekstrak
                        data.append(landmarks)
                        labels.append(label_name)
                    else:
                        print(f"WARNING: No landmarks extracted for {img_path}. Skipping.")
                except Exception as e:
                    print(f"ERROR: Failed to extract landmarks from {img_path}. Error: {e}")
                    continue # Lanjutkan ke label berikutnya
        else:
            print(f"WARNING: Skipping non-directory item in root: {label_path}")


    if not data:
        print(f"WARNING: No data (landmarks) was processed for {sign_language_type}.")


    columns = [f'landmark_{i}_{coord}' for i in range(42) for coord in ['x', 'y', 'z']] 

    # Buat DataFrame
    df = pd.DataFrame(data, columns=columns)
    df['label'] = labels
    # Tambahkan kolom sign_language_type untuk referensi jika diperlukan di kemudian hari
    df['sign_language_type'] = sign_language_type

    print(f"DEBUG: Finished load_and_process_dataset for {sign_language_type}. DataFrame shape: {df.shape}")
    return df

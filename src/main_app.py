# 


import streamlit as st
import sys, os
from PIL import Image
import numpy as np

# Tambahkan folder 'src' agar modul dapat diakses
sys.path.append(os.path.dirname(__file__))

from preprocessing import load_dataset, process_input_image
from eigenface import compute_eigenfaces
from distance import find_nearest_face

st.set_page_config(page_title="Eigenface Recognition", layout="wide")
st.title("Aplikasi Pengenalan Wajah dengan Eigenface")

# Input dataset dan file gambar
dataset_dir = st.text_input("Masukkan path folder dataset:")
uploaded_file = st.file_uploader("Unggah gambar wajah (jpg/png):", type=["jpg","jpeg","png"])

if uploaded_file:
    # Tampilkan preview warna dan grayscale
    img_color = Image.open(uploaded_file)
    img_gray = img_color.convert("L")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_color, caption="Preview Warna", use_container_width=True)
    with c2:
        st.image(img_gray, caption="Preview Grayscale", use_container_width=True)

# Tambahkan slider threshold
threshold = st.slider("Pilih Threshold Euclidean", min_value=0.0, max_value=5000.0, value=800.0, step=50.0)

if st.button("Proses Pencocokan"):
    # Validasi dasar
    if not dataset_dir or not os.path.isdir(dataset_dir):
        st.error("Folder dataset tidak valid.")
        st.stop()
    if not uploaded_file:
        st.error("Mohon unggah gambar untuk pengujian.")
        st.stop()

    # 1. Load dataset dan validasi isi
    A, img_paths, img_size = load_dataset(dataset_dir)
    st.info(f"Dataset berisi {len(img_paths)} gambar. Ukuran tiap gambar: {img_size}")
    if len(img_paths) < 2:
        st.error("Dataset harus berisi minimal 2 gambar untuk PCA.")
        st.stop()

    # 1a. Cek apakah file upload sama dengan salah satu file dataset
    upload_name = uploaded_file.name
    match_names = [os.path.basename(p) for p in img_paths]
    if upload_name in match_names:
        st.success(f"Gambar '{upload_name}' juga ada di dataset.")
    else:
        st.warning(f"Gambar '{upload_name}' tidak ditemukan di dataset. Pastikan nama dan lokasi file.")

    # 2. Hitung eigenfaces dan weights dataset
    mean_face, eigenfaces, weights_train, eigenvalues = compute_eigenfaces(A)

    # Tampilkan detail training (eigenvalues, variance, faces)
    with st.expander("Detail Training - Eigenfaces & Variance"):
        total_var = sum(eigenvalues)
        var_ratio = [ev/total_var for ev in eigenvalues]
        st.write("**Eigenvalues (descending):**", [round(ev,2) for ev in eigenvalues])
        st.write("**Explained Variance Ratio:**", [round(vr,2) for vr in var_ratio])
        st.bar_chart(var_ratio)
        # Tampilkan mean face + beberapa eigenfaces
        cols = st.columns(min(len(eigenfaces[0]),5)+1)
        cols[0].image(mean_face.reshape(img_size[::-1]), clamp=True, channels="L", caption="Mean Face")
        for i, u in enumerate(eigenfaces.T[:5], start=1):
            cols[i].image(u.reshape(img_size[::-1]), clamp=True, channels="L", caption=f"Eigenface {i}")

    # 3. Proses input: konversi ke grayscale, flatten, projeksi
    test_vec, weights_test = process_input_image(uploaded_file, mean_face, eigenfaces, img_size)

    # Debugging: tampilkan beberapa statistik vector
    with st.expander("Debug Input Vector"):
        arr = test_vec.flatten()
        st.write(f"Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.2f}")
        st.line_chart(arr[:100])  # contoh plot 100 pixel pertama

    # 4. Cari wajah terdekat dengan threshold dinamis
    match_path, dist, is_match = find_nearest_face(weights_train, weights_test, img_paths, threshold=threshold)

    # 5. Tampilkan hasil pencocokan
    st.subheader("Hasil Pencocokan")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_color, caption="Input (Warna)")
        st.image(img_gray, caption="Input (Grayscale)")
    with c2:
        st.image(Image.open(match_path), caption=f"Terdekat: {os.path.basename(match_path)}")
    st.write(f"Jarak Euclidean: **{dist:.2f}** (Threshold: {threshold:.1f})")
    st.write(f"Status: **{'Cocok' if is_match else 'Tidak Cocok'}**")
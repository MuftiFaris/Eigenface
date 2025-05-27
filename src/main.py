import streamlit as st
import sys, os
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(__file__))

from preprocessing import load_dataset, process_input_image
from eigenface import compute_eigenfaces
from distance import find_nearest_face

st.set_page_config(page_title="Eigenface Recognition", layout="wide")
st.title("Aplikasi Pengenalan Wajah dengan Eigenface")

dataset_dir = st.text_input("Masukkan path folder dataset:")

if dataset_dir:
    if os.path.isdir(dataset_dir):
        valid_ext = ('.jpg', '.jpeg', '.png')
        count = len([f for f in os.listdir(dataset_dir) if f.lower().endswith(valid_ext)])
        st.success(f"✅ Ditemukan {count} file gambar.")
    else:
        st.error("❌ Path yang dimasukkan bukan folder yang valid.")

uploaded_file = st.file_uploader("Unggah gambar wajah (jpg/png):", type=["jpg","jpeg","png"])

threshold = st.slider("Pilih Threshold Euclidean", min_value=0.0, max_value=5000.0, value=800.0, step=50.0)

if st.button("Proses Pencocokan"):
    if not dataset_dir or not os.path.isdir(dataset_dir):
        st.error("Folder dataset tidak valid.")
        st.stop()
    if not uploaded_file:
        st.error("Mohon unggah gambar untuk pengujian.")
        st.stop()

    A, img_paths, img_size = load_dataset(dataset_dir)
    st.info(f"Dataset berisi {len(img_paths)} gambar. Ukuran tiap gambar: {img_size}")
    if len(img_paths) < 2:
        st.error("Dataset harus berisi minimal 2 gambar untuk PCA.")
        st.stop()

    upload_name = uploaded_file.name
    match_names = [os.path.basename(p) for p in img_paths]
    if upload_name in match_names:
        st.success(f"Gambar '{upload_name}' juga ada di dataset.")
    else:
        st.warning(f"Gambar '{upload_name}' tidak ditemukan di dataset. Pastikan nama dan lokasi file.")

    mean_face, eigenfaces, weights_train, eigenvalues = compute_eigenfaces(A)

    with st.expander("Detail Training - Eigenfaces & Variance"):
        total_var = sum(eigenvalues)
        var_ratio = [ev / total_var if total_var != 0 else 0 for ev in eigenvalues]

        st.write("**Eigenvalues (descending):**", [round(ev,2) for ev in eigenvalues])
        st.write("**Explained Variance Ratio:**", [round(vr,2) for vr in var_ratio])

        st.write("**Bar Chart Variance Ratio:**")
        st.bar_chart(var_ratio)
        st.write("**Line Chart Variance Ratio:**")
        st.line_chart(var_ratio)

        cols = st.columns(min(eigenfaces.shape[1], 5) + 1)
        mf = mean_face.reshape(img_size[::-1])
        mf_norm = (mf - mf.min()) / (mf.max() - mf.min()) if mf.max() != mf.min() else mf
        cols[0].image(mf_norm, caption="Mean Face", use_container_width=True)
        for i, u in enumerate(eigenfaces.T[:5], start=1):
            ef = u.reshape(img_size[::-1])
            ef_norm = (ef - ef.min()) / (ef.max() - ef.min()) if ef.max() != ef.min() else ef
            cols[i].image(ef_norm, caption=f"Eigenface {i}", use_container_width=True)

    test_vec, weights_test = process_input_image(uploaded_file, mean_face, eigenfaces, img_size)

    with st.expander("Debug Input Vector"):
        arr = test_vec.flatten().tolist()
        n = len(arr)
        min_val = min(arr) if n else 0
        max_val = max(arr) if n else 0
        mean_val = sum(arr) / n if n else 0
        var_val = sum((x - mean_val) ** 2 for x in arr) / n if n else 0

        st.write(f"Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}, Variance: {var_val:.4f}")
        st.line_chart(arr[:100])

    match_path, dist, is_match = find_nearest_face(weights_train, weights_test, img_paths, threshold=threshold)

    st.subheader("Hasil Pencocokan")
    c1, c2 = st.columns(2)
    with c1:
        st.image(Image.open(uploaded_file), caption="Input (Warna)")
        st.image(Image.open(uploaded_file).convert("L"), caption="Input (Grayscale)")
    with c2:
        st.image(Image.open(match_path), caption=f"Terdekat: {os.path.basename(match_path)}")
    st.write(f"Jarak Euclidean: **{dist:.2f}** (Threshold: {threshold:.1f})")
    st.write(f"Status: **{'Cocok' if is_match else 'Tidak Cocok'}**")

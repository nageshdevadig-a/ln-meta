import streamlit as st
import numpy as np
import cv2
from PIL import Image
from src.model_arch import MobileNetUNet
from src.processor import process_grid_image, stitch_grid

# Page Config
st.set_page_config(layout="wide", page_title="Camelyon16 Pathology AI")


# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Ensure your .ckpt file is in the checkpoints folder
    path = "checkpoints/best_hybrid_unet.ckpt"
    model = MobileNetUNet.load_from_checkpoint(path, map_location='cpu')
    model.eval()
    return model


model = load_model()

# --- SIDEBAR ---
st.sidebar.header("Settings")
alpha_val = st.sidebar.slider("Heatmap Intensity", 0.0, 1.0, 0.4)
st.sidebar.info("Model: MobileNet-UNet\nInput: 224x224 ")

# --- MAIN UI ---
st.title(" Camelyon Patch-Based Lymph Node metastases Analysis")
st.write("Upload a 4x4 Grid Image (Total size 384x384 pixels).")

uploaded_file = st.file_uploader("Choose a Grid PNG...", type=["png"])

if uploaded_file and model:
    # 1. Load Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    grid_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)

    # 2. Run Processing
    with st.spinner('Analyzing Tissue Patches...'):
        patches, heatmaps, overlays = process_grid_image(grid_img, model, alpha=alpha_val)

        # Stitch individual overlays to create the global view
        global_overlay = stitch_grid(overlays)

    # 3. Display Results in Tabs
    tab1, tab2 = st.tabs(["🌐 Global Overview", "🔍 Individual Patch Analysis"])

    with tab1:
        st.subheader("Global Clinical Mapping")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.image(grid_img, caption="Original Tissue Grid", use_container_width=True)
        with col_g2:
            st.image(global_overlay, caption="Global heatmap Overlay", use_container_width=True)

    with tab2:
        st.subheader("Local Patch Deep Dive")

        col1, col2, col3 = st.columns(3)  # equal width columns

        with col1:
            st.markdown("**Original Tissue**")
     

        with col2:
            st.markdown("**Probability Heatmap**")
       

        with col3:
            st.markdown("**Blended Overlay**")
           

        # Loop through all 16 patches
        for i in range(16):
            with st.expander(f"PATCH #{i + 1} Detailed View"):
                c1, c2, c3 = st.columns(3)

                c1.image(patches[i], caption="Raw Patch (96x96)", use_container_width=True)
                c2.image(heatmaps[i], caption="Tumor Probability mask", use_container_width=True)
                c3.image(overlays[i], caption="Resulting Overlay", use_container_width=True)

else:
    if not model:
        st.error("Model not found! Check your checkpoints folder.")
    else:
        st.info("Please upload a 4x4 patch grid to begin.")
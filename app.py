import streamlit as st
import numpy as np
import cv2
from PIL import Image
from src.processor import process_grid_image, stitch_grid, slice_mask_grid, calculate_dice

# Page Config
st.set_page_config(layout="wide", page_title="Camelyon16 Pathology AI")

# --- MODEL LOADING ---
from src.model_arch import MobileNetUNet


@st.cache_resource
def load_model():
    # Ensure your .ckpt file is in the checkpoints folder
    path = "checkpoints/best_hybrid_unet.ckpt"
    model = MobileNetUNet.load_from_checkpoint(path, map_location='cpu')
    model.eval()
    return model


# from src.model_arch import ResNetUNet
# @st.cache_resource
# def load_model():
#     path = "checkpoints/best_standard_unet.ckpt"
#     model = ResNetUNet.load_from_checkpoint(path, map_location="cpu")
#     model.eval()
#     return model

model = load_model()

# --- SIDEBAR ---
st.sidebar.header("Settings")
alpha_val = st.sidebar.slider("Heatmap Intensity", 0.0, 1.0, 0.4)
st.sidebar.info("Model: MobileNet-UNet\nInput: 224x224 ")

# --- MAIN UI ---
st.title(" Camelyon Patch-Based Lymph Node metastases Analysis")
st.write("Upload a 4x4 Grid Image (Total size 384x384 pixels).")

# Modified uploader section to include Ground Truth
col_u1, col_u2 = st.columns(2)
with col_u1:
    uploaded_file = st.file_uploader("Choose a Grid PNG...", type=["png"])
with col_u2:
    uploaded_mask = st.file_uploader("Choose Ground Truth Mask Grid (Optional)", type=["png"])

if uploaded_file and model:
    # 1. Load Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    grid_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)

    # 2. Run Processing
    with st.spinner('AI analyzing tissue...'):
        patches, heatmaps, overlays = process_grid_image(grid_img, model, alpha=alpha_val)

        # FIXED LOGIC: Turn Heatmap into Binary correctly
        pred_binary_masks = []
        for h in heatmaps:
            # Heatmaps are RGB from cv2.applyColorMap
            # In JET colormap, Red (Tumor) has high values in the RED channel (Index 0)
            red_channel = h[:, :, 0]
            _, binary = cv2.threshold(red_channel, 127, 255, cv2.THRESH_BINARY)
            pred_binary_masks.append(binary)

        global_overlay = stitch_grid(overlays)

    # Optional: Load Ground Truth if provided
    gt_masks = None
    if uploaded_mask:
        mask_bytes = np.asarray(bytearray(uploaded_mask.read()), dtype=np.uint8)
        gt_grid = cv2.imdecode(mask_bytes, cv2.IMREAD_COLOR)
        gt_masks = slice_mask_grid(gt_grid)

    # --- GLOBAL STATISTICS & CLASSIFICATION ---
    st.divider()
    # Logic: If any patch has more than a tiny amount of predicted tumor, flag as CANCER FOUND
    tumor_pixels_per_patch = [np.sum(m > 127) for m in pred_binary_masks]
    affected_patches = sum([1 for p in tumor_pixels_per_patch if p > 50])  # threshold to avoid noise
    cancer_status = "🔴 CANCER DETECTED" if affected_patches > 0 else "🟢 CLEAR"

    # Calculate global tumor load (percentage of grid covered by tumor)
    total_pixels = 384 * 384
    total_tumor_pixels = sum(tumor_pixels_per_patch)
    tumor_load = (total_tumor_pixels / total_pixels) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Diagnosis", cancer_status)
    c2.metric("Tumor Load", f"{tumor_load:.2f}% of Grid")
    if gt_masks:
        global_dice = np.mean([calculate_dice(pred_binary_masks[i], gt_masks[i]) for i in range(16)])
        c3.metric("Global Accuracy (Dice)", f"{global_dice:.2%}")

    # 3. Display Results in Tabs
    tab1, tab2 = st.tabs(["Global Overview", "Individual Patch Analysis"])

    with tab1:
        st.subheader("Global Clinical Mapping")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.image(grid_img, caption="Original Tissue Grid", use_container_width=True)
        with col_g2:
            st.image(global_overlay, caption="Global heatmap Overlay", use_container_width=True)

    with tab2:
        st.subheader("Local Patch Deep Dive & Boundary Validation")
        # Header row
        h_cols = st.columns(4 if gt_masks else 3)
        h_cols[0].markdown("**Original Tissue**")
        h_cols[1].markdown("**AI Heatmap**")
        if gt_masks:
            h_cols[2].markdown("**GT Boundary Overlay**")  # New Column
            h_cols[3].markdown("**Validation Metrics**")
        else:
            h_cols[2].markdown("**Blended Overlay**")

        for i in range(16):
            with st.expander(f"PATCH #{i + 1} Analysis"):
                cols = st.columns(4 if gt_masks else 3)
                cols[0].image(patches[i], use_container_width=True)
                cols[1].image(heatmaps[i], use_container_width=True)

                if gt_masks:
                    # Column 3: Show GT boundaries (Green) on the AI Overlay
                    # This lets you see the "Exact Difference"
                    from src.processor import get_gt_boundary_overlay

                    gt_boundary_view = get_gt_boundary_overlay(overlays[i], gt_masks[i])
                    cols[2].image(gt_boundary_view, caption="Green Line = Ground Truth", use_container_width=True)

                    # Column 4: Metrics with FIXED Dice
                    dice_score = calculate_dice(heatmaps[i], gt_masks[i])

                    color = "green" if dice_score > 0.75 else "orange" if dice_score > 0.3 else "red"
                    status = "✅ Match" if dice_score > 0.75 else "⚠️ Partial" if dice_score > 0.3 else "❌ Mismatch"

                    cols[3].markdown(f"**Dice:** <span style='color:{color}'>{dice_score:.4f}</span>",
                                     unsafe_allow_html=True)
                    cols[3].write(f"Status: {status}")
                else:
                    cols[2].image(overlays[i], use_container_width=True)

else:
    if not model:
        st.error("Model not found! Check your checkpoints folder.")
    else:
        st.info("Please upload a 4x4 patch grid to begin.")
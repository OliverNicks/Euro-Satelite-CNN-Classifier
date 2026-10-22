"""EuroSAT Satellite Land Cover Classification - Interactive Streamlit Web App.

Run locally:
    streamlit run app/app.py
"""

from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F

# Add project root to sys.path if not present
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset import get_inference_transforms
from src.model import CLASS_DESCRIPTIONS, CLASS_NAMES, IDX_TO_CLASS, SateliteCNN, load_model

# Page configuration
st.set_page_config(
    page_title="EuroSAT Satellite Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #4F46E5;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #10B981;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Satellite CNN weights...")
def get_cached_model(weights_path: str) -> Tuple[SateliteCNN, torch.device]:
    """Cache model in memory for instantaneous CPU inference."""
    device = torch.device("cpu")
    model = load_model(weights_path=weights_path, device=device)
    return model, device


def run_inference(image: Image.Image, model: SateliteCNN, device: torch.device) -> pd.DataFrame:
    """Preprocess PIL image and compute class probabilities."""
    transform = get_inference_transforms()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    df = pd.DataFrame({
        "Class": [IDX_TO_CLASS[i] for i in range(len(CLASS_NAMES))],
        "Probability": probs.cpu().numpy() * 100,
    }).sort_values(by="Probability", ascending=False).reset_index(drop=True)

    return df


def main() -> None:
    # Resolve default weights path
    default_weights = os.path.join(ROOT_DIR, "app", "satellite_model_9715.pth")

    # Sidebar: Project Info & Sample Selector
    with st.sidebar:
        st.image("https://img.shields.io/badge/Model-SateliteCNN-blue?style=for-the-badge&logo=pytorch", use_container_width=False)
        st.image("https://img.shields.io/badge/Accuracy-97.15%25-brightgreen?style=for-the-badge", use_container_width=False)
        st.markdown("### 🛰️ EuroSAT Classifier")
        st.markdown(
            "Deep learning land use and land cover (LULC) classifier trained on "
            "**EuroSAT Sentinel-2 satellite imagery** (13 spectral bands / RGB)."
        )

        st.markdown("---")
        st.markdown("#### ⚡ Quick Test with Samples")
        samples_dir = os.path.join(ROOT_DIR, "samples")
        sample_options = ["(Upload Custom Image)"]
        if os.path.exists(samples_dir):
            sample_files = [f for f in sorted(os.listdir(samples_dir)) if f.endswith(".jpg")]
            sample_options += sample_files

        selected_sample = st.selectbox("Choose a sample satellite image:", sample_options)

        st.markdown("---")
        with st.expander("📖 Class Guide (10 Categories)"):
            for cls_name, desc in CLASS_DESCRIPTIONS.items():
                st.markdown(f"**{cls_name}**: {desc}")

        st.markdown("---")
        st.caption("Built with PyTorch & Streamlit | 97.15% Validation Accuracy")

    # Main Tabs
    tab1, tab2 = st.tabs(["🔍 Live Image Classification", "📊 Model Architecture & Benchmark Insights"])

    # Load Model
    if not os.path.exists(default_weights):
        st.error(f"Model weights file not found at `{default_weights}`. Please check repository setup.")
        return

    model, device = get_cached_model(default_weights)

    with tab1:
        st.markdown('<div class="main-header">🛰️ Satellite Land Cover Classifier</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Upload a Sentinel-2 satellite patch or select a sample to identify land cover category.</div>',
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([1, 1.4], gap="large")

        active_image: Optional[Image.Image] = None
        active_source_label = ""

        with col_left:
            st.markdown("#### 1. Input Image")
            uploaded_file = st.file_uploader(
                "Upload a satellite image (.jpg, .png)",
                type=["jpg", "jpeg", "png"],
                help="Recommended resolution: 64x64 or higher RGB satellite image.",
            )

            if uploaded_file is not None:
                active_image = Image.open(uploaded_file)
                active_source_label = uploaded_file.name
            elif selected_sample != "(Upload Custom Image)":
                sample_path = os.path.join(samples_dir, selected_sample)
                active_image = Image.open(sample_path)
                active_source_label = selected_sample

            if active_image is not None:
                st.image(active_image, caption=f"Selected: {active_source_label}", use_container_width=True)
            else:
                st.info("👆 Upload an image or select one from the sidebar dropdown to run inference.")

        with col_right:
            st.markdown("#### 2. Prediction Analysis")
            if active_image is not None:
                with st.spinner("Analyzing spectral patterns..."):
                    pred_df = run_inference(active_image, model, device)

                top_class = pred_df.iloc[0]["Class"]
                top_conf = pred_df.iloc[0]["Probability"]
                top_desc = CLASS_DESCRIPTIONS.get(top_class, "")

                # Metric highlight
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #4F46E5; font-weight: 600;">PREDICTED LAND COVER</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #111827;">{top_class}</div>
                        <div style="margin-top: 0.4rem;">
                            <span class="badge">{top_conf:.2f}% Confidence</span>
                        </div>
                        <p style="margin-top: 0.8rem; font-size: 0.95rem; color: #374151;">{top_desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Probabilities chart
                st.markdown("##### Class Probability Distribution")
                chart_df = pred_df.head(5).set_index("Class")
                st.bar_chart(chart_df["Probability"], color="#4F46E5", use_container_width=True)

                # Detailed Table
                with st.expander("📋 View All 10 Class Probabilities"):
                    formatted_df = pred_df.copy()
                    formatted_df["Probability"] = formatted_df["Probability"].map("{:.2f}%".format)
                    st.dataframe(formatted_df, use_container_width=True, hide_index=True)
            else:
                st.write("Awaiting image input...")

    with tab2:
        st.markdown("### 📊 Model Architecture & Training Evaluation")
        st.markdown(
            """
            The **SateliteCNN** model consists of **4 dual-convolutional blocks (8 Conv layers total)**,
            each paired with **Batch Normalization**, **ReLU activations**, **Dropout regularization (0.1 - 0.3)**,
            and **Adaptive Average Pooling** feeding a dense classification head.
            """
        )

        assets_dir = os.path.join(ROOT_DIR, "assets")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Training & Validation Performance")
            tc_path = os.path.join(assets_dir, "training_curves.png")
            if os.path.exists(tc_path):
                st.image(tc_path, caption="Cross-Entropy Loss & Accuracy across Epochs (97.15% peak)", use_container_width=True)
            else:
                st.info("Training curves plot not found in assets.")

        with col_b:
            st.markdown("#### Test Set Confusion Matrix")
            cm_path = os.path.join(assets_dir, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix over 2,000 Validation Images", use_container_width=True)
            else:
                st.info("Confusion matrix plot not found in assets.")

        st.markdown("---")
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("#### EuroSAT Sample Imagery")
            rs_path = os.path.join(assets_dir, "random_samples.png")
            if os.path.exists(rs_path):
                st.image(rs_path, caption="Random Sentinel-2 64x64 Patches", use_container_width=True)

        with col_d:
            st.markdown("#### Average Brightness per Category")
            br_path = os.path.join(assets_dir, "brightness_per_class.png")
            if os.path.exists(br_path):
                st.image(br_path, caption="Mean Spectral Brightness Comparison", use_container_width=True)


if __name__ == "__main__":
    main()

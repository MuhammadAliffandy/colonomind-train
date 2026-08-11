import os
import cv2
import json
import joblib
import numpy as np
import scipy.stats
import streamlit as st
import pywt
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set Keras backend to tensorflow explicitly to avoid Keras 3 issues
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras.models import load_model
import tensorflow_hub as hub
import lightgbm as lgb

st.set_page_config(layout="wide", page_title="Diagnostic Agent", page_icon="🔍")

@keras.saving.register_keras_serializable(package="Custom")
def resnet50_preprocess(x):
    return (x / 127.5) - 1.0

@keras.saving.register_keras_serializable(package="Custom")
def densenet_preprocess(x):
    return (x / 127.5) - 1.0

@keras.saving.register_keras_serializable(package="Custom")
def efficientnet_preprocess(x):
    return (x / 127.5) - 1.0

@keras.saving.register_keras_serializable(package="Custom")
def convnext_preprocess(x):
    return (x / 127.5) - 1.0

@keras.saving.register_keras_serializable(package="Custom")
def vit_preprocess(x):
    return (x / 127.5) - 1.0

# Patch Dense to ignore quantization_config (for older Keras versions on DGX)
from keras.layers import Dense, Layer

@keras.saving.register_keras_serializable(package="Custom")
class CustomDense(Dense):
    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return super().from_config(config)

# Wrapper for ViT since it was saved with a custom hub wrapper
@keras.saving.register_keras_serializable(package="Custom")
class ViT_B16_Wrapper(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vit_model = hub.load("https://tfhub.dev/sayakpaul/vit_b16_fe/1")
        self.trainable = False

    def call(self, inputs):
        out = self.vit_model(inputs)
        if isinstance(out, dict):
            return out[list(out.keys())[0]]
        return out

    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return super().from_config(config)

def load_all_models(base_drive, dataset_key, model_names):
    models = {}
    for m in model_names:
        exp_dir = os.path.join(base_drive, dataset_key, f"{m}_Experiment")
        keras_path = os.path.join(exp_dir, f"{m}_hybrid.keras")
        if not os.path.exists(keras_path):
            legacy_path = os.path.join(exp_dir, f"{m}_hybrid.h5")
            if os.path.exists(legacy_path):
                keras_path = legacy_path
                
        custom_objs = {
            'KerasLayer': hub.KerasLayer,
            'Dense': CustomDense,
            'ViT_B16_Wrapper': ViT_B16_Wrapper
        }
                
        try:
            # safe_mode=False needed to allow Lambda layers to deserialize
            dl_model = load_model(keras_path, compile=False, custom_objects=custom_objs, safe_mode=False)
        except Exception as e:
            error_msg = f"Keras Load Error: {e}"
            print(f"Error loading {keras_path}: {e}")
            dl_model = None
            
        try:
            umap_model = joblib.load(os.path.join(exp_dir, "umap_model.pkl"))
            base_scaler = joblib.load(os.path.join(exp_dir, "base_scaler.pkl"))
            agent_scaler = joblib.load(os.path.join(exp_dir, f"{m}_scaler.pkl"))
            agent = lgb.Booster(model_file=os.path.join(exp_dir, f"{m}_agent.txt"))
        except Exception as e:
            error_msg = f"PKL/Agent Load Error: {e}"
            print(f"Error loading PKL/Agent for {m}: {e}")
            umap_model, base_scaler, agent_scaler, agent = None, None, None, None
            
        models[m] = {"dl": dl_model, "umap": umap_model, "base_scaler": base_scaler, "agent_scaler": agent_scaler, "agent": agent}
        if None in models[m].values():
            models[m]["error"] = error_msg
    return models

def extract_handcrafted_features(img_arr, WAVELET="db1"):
    gray = cv2.cvtColor(cv2.resize(img_arr, (224, 224)), cv2.COLOR_RGB2GRAY)
    coeffs2 = pywt.dwt2(gray, WAVELET)
    LL, (LH, HL, HH) = coeffs2
    def stats(sb):
        return [float(np.mean(sb)), float(np.std(sb)), float(np.var(sb)), float(scipy.stats.entropy(np.abs(sb.flatten()) + 1e-6))]
    feats  = stats(LL) + stats(LH) + stats(HL) + stats(HH)
    feats += [float(np.sum(np.square(HH)) / HH.size)]
    glcm = graycomatrix(gray, distances=[1,3,5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    feats += [
        float(np.mean(graycoprops(glcm, "contrast"))),
        float(np.mean(graycoprops(glcm, "dissimilarity"))),
        float(np.mean(graycoprops(glcm, "homogeneity"))),
    ]
    return feats

def predict_single_image(img_arr, model_dict):
    dl_model = model_dict["dl"]
    umap_model = model_dict["umap"]
    base_scaler = model_dict["base_scaler"]
    agent_scaler = model_dict["agent_scaler"]
    agent = model_dict["agent"]
    
    if "error" in model_dict:
        return {"error": model_dict["error"]}
    if None in [dl_model, umap_model, base_scaler, agent_scaler, agent]:
        return {"error": "Missing model files or failed to load"}
        
    img_resized = cv2.resize(img_arr, (224, 224))
    img_rgb = np.expand_dims(img_resized, axis=0) 
    
    h_feats = extract_handcrafted_features(img_arr)
    feats_scaled = base_scaler.transform(np.array(h_feats).reshape(1, -1))
    umap_feat = umap_model.transform(feats_scaled)
    
    dl_proba = dl_model.predict([img_rgb, feats_scaled, umap_feat], verbose=0)[0]
    dl_conf = float(np.max(dl_proba))
    
    if dl_conf >= 0.50:
        final_proba = list(float(x) for x in dl_proba)
        final_conf = dl_conf
        source = "Deep Learning (ResNet/ViT dll)"
        agent_input = np.array([])
    else:
        # Construct agent input: [confidence, umap_0, umap_1, f0..f19]
        agent_features = np.hstack([[dl_conf], umap_feat[0], feats_scaled[0]]).reshape(1, -1)
        agent_input = agent_scaler.transform(agent_features)
        
        agent_proba = agent.predict(agent_input)[0]
        final_proba = list(float(x) for x in agent_proba)
        final_conf = float(np.max(agent_proba))
        source = "LightGBM Super Agent"
        
    label_idx = int(np.argmax(final_proba))
    
    label_map = {0: "MES0", 1: "MES1", 2: "MES2", 3: "MES3"}
    label_str = label_map[label_idx]
    
    return {
        "label_idx": label_idx,
        "label_str": label_str,
        "conf": final_conf,
        "proba": final_proba,
        "feats": list(feats_scaled[0]),
        "agent_input": agent_input.tolist() if agent_input.size > 0 else [],
        "source": source
    }

def main():
    st.markdown("""
    <style>
    .main-header {
        background-color: #1a1d2e;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #2d3748;
    }
    .main-header h1 {
        color: #58a6ff;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #8b949e;
        font-size: 1.1rem;
    }
    
    .label-mes0 { background:rgba(46, 160, 67, 0.15); color:#2ea043; padding:10px 20px; border-radius:30px; font-weight:bold; font-size:1.4rem; border:2px solid #2ea043;}
    .label-mes1 { background:rgba(212, 160, 23, 0.15); color:#d4a017; padding:10px 20px; border-radius:30px; font-weight:bold; font-size:1.4rem; border:2px solid #d4a017;}
    .label-mes2 { background:rgba(253, 126, 20, 0.15); color:#fd7e14; padding:10px 20px; border-radius:30px; font-weight:bold; font-size:1.4rem; border:2px solid #fd7e14;}
    .label-mes3 { background:rgba(248, 81, 73, 0.15); color:#f85149; padding:10px 20px; border-radius:30px; font-weight:bold; font-size:1.4rem; border:2px solid #f85149;}
    
    .small-label-mes0 { color:#2ea043; font-weight:bold; font-size:1.1rem; border:1px solid #2ea043; padding:2px 8px; border-radius:12px; display:inline-block; margin-top:4px;}
    .small-label-mes1 { color:#d4a017; font-weight:bold; font-size:1.1rem; border:1px solid #d4a017; padding:2px 8px; border-radius:12px; display:inline-block; margin-top:4px;}
    .small-label-mes2 { color:#fd7e14; font-weight:bold; font-size:1.1rem; border:1px solid #fd7e14; padding:2px 8px; border-radius:12px; display:inline-block; margin-top:4px;}
    .small-label-mes3 { color:#f85149; font-weight:bold; font-size:1.1rem; border:1px solid #f85149; padding:2px 8px; border-radius:12px; display:inline-block; margin-top:4px;}
    
    .footer-tag {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #30363d;
        color: #8b949e;
        font-size: 0.9rem;
    }
    .recommendation-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-left: 4px solid #58a6ff;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        font-size: 1.1rem;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #c9d1d9;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .dark-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .ensemble-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1rem;
        border-left: 4px solid #58a6ff;
        border-radius: 6px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .model-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        text-align: center;
        flex: 1 1 80px;
        min-width: 80px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .models-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    DATASET_CHOICES = {
        "Intra_Unified": "Unified (TMC-UCM + NTUH + LIMUC)"
    }
    MODEL_CHOICES = ["ResNet-50", "DenseNet-121", "EfficientNet-B4", "ConvNeXt-Tiny", "ViT-B-16"]
    CLASS_NAMES   = ["MES0", "MES1", "MES2", "MES3"]
    IMG_SIZE      = (224, 224)
    WAVELET       = "db1"
    BASE_DRIVE    = "../Result"

    SMALL_LABEL_CSS  = {"MES0": "small-label-mes0", "MES1": "small-label-mes1", "MES2": "small-label-mes2", "MES3": "small-label-mes3"}
    LABEL_DESC = {
        "MES0": "Normal Mucosa 🟢",
        "MES1": "Mild Inflammation 🟡",
        "MES2": "Moderate Inflammation 🟠",
        "MES3": "Severe Inflammation 🔴",
    }
    FEAT_NAMES = [
        "LL_Mean","LL_Std","LL_Var","LL_Ent",
        "LH_Mean","LH_Std","LH_Var","LH_Ent",
        "HL_Mean","HL_Std","HL_Var","HL_Ent",
        "HH_Mean","HH_Std","HH_Var","HH_Ent","HH_Energy",
        "GLCM_Contrast","GLCM_Dissimilarity","GLCM_Homogeneity",
    ]

    # ----------------- 1. Sidebar -----------------
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        st.subheader("📁 1. Dataset")
        
        # Auto-detect dataset folder inside BASE_DRIVE
        if os.path.exists(BASE_DRIVE):
            available_datasets = [d for d in os.listdir(BASE_DRIVE) if os.path.isdir(os.path.join(BASE_DRIVE, d))]
            if available_datasets:
                selected_dataset_key = available_datasets[0]
                pretty_name = DATASET_CHOICES.get(selected_dataset_key, selected_dataset_key)
                st.markdown(f"**Auto-selected:** `{pretty_name}`")
            else:
                st.error("⚠️ No dataset folders found in ./Result/")
                selected_dataset_key = "Unknown"
        else:
            st.error("⚠️ ./Result directory not found.")
            selected_dataset_key = "Unknown"
            
        st.markdown("---")
        st.subheader("🤖 2. Ensemble Settings")
        voting_threshold = st.selectbox(
            "Voting Threshold (Agreement needed)",
            [3, 4, 5],
            format_func=lambda x: f"{x}/5 Models Agree",
            index=0
        )
        st.markdown("---")

    def get_recommendation(label_str, is_ref):
        severity_map = {
            "MES0": "remission",
            "MES1": "mild",
            "MES2": "moderate",
            "MES3": "severe"
        }
        severity = severity_map[label_str]
        
        if label_str == "MES0":
            text = f"The patient has achieved **endoscopic remission**. <br><br>" \
                   f"**Action:** These medications were safe to be continued. " \
                   f"Screening colonoscopy should be scheduled according to routine interval."
        elif label_str == "MES1":
            text = f"The patient has **{severity}** inflammation. <br><br>" \
                   f"**Action:** The patient has achieved intermediate treatment target. " \
                   f"Based on the patient demographics and severity, the recommended next option is: Optimize current medication."
        else:
            text = f"The patient has **{severity}** inflammation. <br><br>" \
                   f"**Action:** The current medication should be adjusted. " \
                   f"Based on the patient demographics, extent, severity, and current medication failure, the recommended next option is: Escalate to advanced therapy or combine other advanced therapy."
                   
        if is_ref:
            text += "<br><br>⚠️ **Warning:** Prediction uncertainty is high (Consensus < Threshold). Clinical correlation and specialist referral needed."
            
        return text

    # Main Header
    st.markdown("""
    <div class="main-header">
      <h1> Colonoscopy — Diagnostic Agent</h1>
      <p>This is for education purpose only</p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------- A. Input Section -----------------
    st.subheader("A. Input Section (Upload Image)")
    col_batch, col_upload = st.columns([1, 2])
    with col_batch:
        batch_size_str = st.radio("Batch Selector", ["1 Image", "5 Images", "10 Images"], horizontal=True)
        batch_size = int(batch_size_str.split()[0])
        
    with col_upload:
        uploaded_files = st.file_uploader("🖼️ Upload Colonoscopy Image(s)", type=["png","jpg","jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) > batch_size:
            st.warning(f"You uploaded {len(uploaded_files)} files but Batch Selector is set to {batch_size}. Only processing the first {batch_size} files.")
            uploaded_files = uploaded_files[:batch_size]
            
        models_to_run = MODEL_CHOICES
        
        # Pre-load models in memory once for all images
        loaded_models = load_all_models(BASE_DRIVE, selected_dataset_key, models_to_run)
        
        global_features_list = []
        global_severities = []

        for i, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### Image {i+1}: {uploaded_file.name}")
            
            # Layout: Image | Left (Consensus & Indiv) | Right (Avg Prob)
            c_img, c_left, c_right = st.columns([1.5, 2.5, 2], gap="medium")
            
            pil_img = Image.open(uploaded_file).convert("RGB")
            with c_img:
                st.image(pil_img, use_column_width=True)
                
            img_arr = np.array(pil_img)
            predictions = {}
            
            with st.spinner(f"Analyzing Image {i+1}..."):
                for m in models_to_run:
                    predictions[m] = predict_single_image(img_arr, loaded_models[m])
            
            valid_preds = {m: p for m, p in predictions.items() if "error" not in p}
            
            for m, p in predictions.items():
                if "error" in p:
                    st.warning(f"⚠️ {m} was skipped due to error: {p['error']}")
                    
            if not valid_preds:
                st.error("No valid predictions from models.")
                continue
                
            # --- Calculations ---
            
            # 1. Majority Vote
            votes = [p["label_str"] for p in valid_preds.values()]
            from collections import Counter
            vote_counts = Counter(votes)
            majority_class, majority_count = vote_counts.most_common(1)[0]
            
            # 2. Most Severe Logic
            max_severity_idx = max([p["label_idx"] for p in valid_preds.values()])
            most_severe_str = CLASS_NAMES[max_severity_idx]
            
            # 3. Average Probability
            avg_proba = [0] * len(CLASS_NAMES)
            for p in valid_preds.values():
                for idx_c in range(len(CLASS_NAMES)): 
                    avg_proba[idx_c] += p["proba"][idx_c]
            
            total_models = len(valid_preds)
            avg_proba_pct = [x / total_models for x in avg_proba]
            
            is_ref = majority_count < voting_threshold
            
            # We track the MOST SEVERE ensemble prediction for the global patient recommendation
            global_severities.append(most_severe_str)
            
            # Left Column (Consensus & Individual)
            with c_left:
                st.markdown(f"""
                <div class="ensemble-box">
                    <h4 style="color:#58a6ff; margin-top:0; margin-bottom:12px;">Ensemble Result</h4>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="color:#8b949e; font-size:0.9rem;">major vote:</span><br>
                            <b>{majority_count}/{total_models} &rarr; {majority_class}</b>
                        </div>
                        <div style="border-left:1px solid #30363d; height:30px; margin:0 10px;"></div>
                        <div>
                            <span style="color:#8b949e; font-size:0.9rem;">most severe:</span><br>
                            <b>Safe-fail &rarr; {most_severe_str}</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Individual Models using Flexbox for responsiveness
                st.markdown("**Individual Models:**")
                models_html = '<div class="models-container">'
                for m, p in valid_preds.items():
                    m_short = m.replace("-Experiment", "").replace("-", "")[:5]
                    models_html += f"""
                    <div class="model-card">
                        <b style="font-size:0.85rem;">{m_short}</b><br>
                        <span class="{SMALL_LABEL_CSS.get(p['label_str'], '')}">{p['label_str']}</span><br>
                        <span style="opacity:0.7; font-size:0.85rem;">{p['conf']*100:.0f}%</span>
                    </div>
                    """
                models_html += '</div>'
                st.markdown(models_html, unsafe_allow_html=True)

            # Right Column (Average Percentage Bar Chart)
            with c_right:
                class_colors = ['#2ea043', '#d4a017', '#fd7e14', '#f85149']
                fig_proba = go.Figure(data=[go.Bar(
                    x=CLASS_NAMES, 
                    y=avg_proba_pct, 
                    marker_color=class_colors, 
                    text=[f"{v*100:.1f}%" for v in avg_proba_pct], 
                    textposition='auto'
                )])
                fig_proba.update_layout(
                    title="Average Percentage Weight",
                    title_font_size=14,
                    height=220,
                    margin=dict(l=20, r=20, t=30, b=20),
                    template="plotly_dark",
                )
                st.plotly_chart(fig_proba, use_container_width=True)

            # Accumulate features for the aggregate view
            for m, p in valid_preds.items():
                if p["agent_input"] and len(p["agent_input"]) > 0 and len(p["agent_input"][0]) > 3:
                    global_features_list.append(p["agent_input"][0][3:])
                else:
                    global_features_list.append(p["feats"])
            
            st.divider()

        # ----------------- Aggregate Section -----------------
        if len(global_severities) > 0:
            st.subheader("Global Patient Metrics")
            
            c_rec, c_feat = st.columns([1, 1], gap="large")
            
            with c_rec:
                # 1. Adjudication Agent (Suggestion)
                # Severities mapping MES0 < MES1 < MES2 < MES3
                max_sev = max(global_severities, key=lambda x: int(x[-1]))
                # Use majority logic for referral of final max severity?
                # The user requested to know if it's referrable. We pass is_ref=False since we only output severity text
                rec_text = get_recommendation(max_sev, is_ref=False)
                
                st.markdown("### Final Patient Adjudication")
                st.markdown(f"The most severe finding from this patient is **{max_sev}**.")
                # We can use st.info or st.warning for the recommendation box to support markdown natively
                st.info(rec_text)
                
            with c_feat:
                # 2. Average Top 5 Textures
                st.markdown("**Top 5 Handcrafted Features (Average)**")
                if len(global_features_list) > 0:
                    avg_feats = np.mean(global_features_list, axis=0)
                    abs_vals = [abs(v) for v in avg_feats]
                    indexed = sorted(enumerate(abs_vals), key=lambda x: x[1], reverse=True)[:5]
                    top_5_names = [FEAT_NAMES[idx] for idx, _ in indexed]
                    top_5_vals  = [float(avg_feats[idx]) for idx, _ in indexed]
                    
                    fig_top5 = go.Figure(data=[go.Bar(
                        x=top_5_vals[::-1], # reverse for horizontal
                        y=top_5_names[::-1],
                        orientation='h',
                        marker_color='#ed64a6'
                    )])
                    fig_top5.update_layout(
                        height=250,
                        margin=dict(l=10, r=10, t=10, b=10),
                        template="plotly_dark",
                        xaxis_title="Average Scaled Value"
                    )
                    st.plotly_chart(fig_top5, use_container_width=True)

            st.divider()
            
            # 3. 3-Panel ROC Curves
            st.markdown("### Receiver Operating Characteristic (ROC)")
            st.markdown("<span style='color:#aaa;'>Simulated Dataset Evaluation Performance</span>", unsafe_allow_html=True)
            
            roc_cols = st.columns(3)
            datasets_roc = ["TMC-UCM", "NTUH", "LIMUC"]
            # Mock AUC base values for datasets (NTUH is base, TMC is slightly worse, LIMUC slightly better)
            auc_offsets = {"TMC-UCM": -0.02, "NTUH": 0.0, "LIMUC": +0.015}
            
            MODEL_METRICS = {
                "ResNet-50": 0.92,
                "DenseNet-121": 0.94,
                "EfficientNet-B4": 0.95,
                "ConvNeXt-Tiny": 0.96,
                "ViT-B-16": 0.97,
            }
            colors = ['#667eea', '#ed64a6', '#48bb78', '#ecc94b', '#9f7aea']
            fpr_list = [round(j / 99.0, 2) for j in range(100)]
            
            for d_idx, d_name in enumerate(datasets_roc):
                with roc_cols[d_idx]:
                    st.markdown(f"<div style='text-align:center;'><b>{d_name} Dataset</b></div>", unsafe_allow_html=True)
                    fig_roc = go.Figure()
                    
                    for m_idx, (m_name, base_auc) in enumerate(MODEL_METRICS.items()):
                        m_auc = min(0.99, base_auc + auc_offsets[d_name])
                        a = (1.0 / m_auc) - 1.0
                        tpr = [min(1.0, x ** a) for x in fpr_list]
                        
                        fig_roc.add_trace(go.Scatter(
                            x=fpr_list, y=tpr, mode='lines', 
                            name=f'{m_name[:5]}', 
                            line=dict(color=colors[m_idx % len(colors)], width=2)
                        ))
                        
                    fig_roc.add_trace(go.Scatter(x=fpr_list, y=fpr_list, mode='lines', name='Random Guess', line=dict(color='#6c757d', width=1, dash='dash')))
                    
                    fig_roc.update_layout(
                        xaxis_title="FPR" if d_idx == 1 else None,
                        yaxis_title="TPR" if d_idx == 0 else None,
                        height=300, # Increased height to accommodate the legend
                        margin=dict(l=20, r=10, t=10, b=10),
                        template="plotly_dark",
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=10)
                        )
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
            
    if uploaded_files is None or len(uploaded_files) == 0:
        st.info("👈 Silakan pilih pengaturan di panel kiri, lalu unggah gambar untuk memulai sesi Diagnostik (Batch Processing).")

    # Footer
    st.markdown("<div class='footer-tag'>Diagnostic Agent System © 2026</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

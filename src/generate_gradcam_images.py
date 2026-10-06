import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# CONFIG
# ==============================================================================
IMG_SIZE = (256, 256)

@tf.keras.utils.register_keras_serializable()
class ChannelPooling(tf.keras.layers.Layer):
    def __init__(self, pool_type='mean', **kwargs):
        super(ChannelPooling, self).__init__(**kwargs)
        self.pool_type = pool_type
    def call(self, inputs):
        if self.pool_type == 'mean': return tf.reduce_mean(inputs, axis=-1, keepdims=True)
        else: return tf.reduce_max(inputs, axis=-1, keepdims=True)

# ==============================================================================
# PURE TENSORFLOW GRAD-CAM IMPLEMENTATION
# ==============================================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Buat model fungsional baru yang memotong model asli menjadi 2 bagian:
    # 1. Input -> Last Conv Layer
    # 2. Input -> Final Predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # GradientTape untuk merekam gradien dari kelas target terhadap feature map
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Hitung gradien
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Global average pooling gradien
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Kalikan feature map dengan bobot gradien (importance)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU (buang nilai negatif) dan normalisasi ke 0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, save_path, alpha=0.4):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap ke ukuran gambar asli
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Konversi heatmap ke RGB menggunakan colormap Jet (seperti di paper medis)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Overlay (Gabungkan gambar asli dengan heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Simpan hasil
    superimposed_img.save(save_path)
    print(f"✅ Grad-CAM tersimpan di: {save_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, help="Path ke folder gambar contoh (misal: gambar tes dari MES0, 1, 2, 3)")
    ap.add_argument("--model_path", required=True, help="Path ke file model (.h5)")
    ap.add_argument("--conv_layer", default=None, help="Nama layer konvolusi terakhir. Jika kosong, script akan mencarinya otomatis.")
    ap.add_argument("--out_dir", default="../Result/GradCAM_Outputs", help="Folder output untuk menyimpan foto Grad-CAM")
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model dari {args.model_path}...")
    model = load_model(args.model_path, compile=False)
    
    # Cari nama layer konvolusi terakhir jika tidak disebutkan
    last_conv_name = args.conv_layer
    if not last_conv_name:
        for layer in reversed(model.layers):
            # Mencari layer 3D terakhir (biasanya Conv2D atau Activation sebelum GlobalAveragePooling)
            if len(layer.output_shape) == 4:
                last_conv_name = layer.name
                break
    print(f"Menggunakan layer: {last_conv_name} sebagai target ekstraksi Grad-CAM")
    
    # Proses semua gambar di dalam folder
    valid_ext = [".jpg", ".jpeg", ".png"]
    for fname in os.listdir(args.image_dir):
        if not any(fname.lower().endswith(ext) for ext in valid_ext):
            continue
            
        img_path = os.path.join(args.image_dir, fname)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocessing (samakan dengan pipeline Anda)
        img_array = np.expand_dims(img, axis=0) / 255.0
        
        # Jika model butuh input fitur tabular (Deep Fusion), berikan array dummy sementara
        # karena Grad-CAM hanya fokus pada bobot spasial dari gambar
        if isinstance(model.input, list):
            dummy_feat = np.zeros((1, 28)) # Dummy 28 fitur klinis/tekstur
            dummy_umap = np.zeros((1, 2))  # Dummy 2 UMAP
            inputs = [img_array, dummy_feat, dummy_umap]
        else:
            inputs = img_array
            
        # Prediksi probabilitas
        preds = model.predict(inputs, verbose=0)
        pred_class = np.argmax(preds[0])
        confidence = np.max(preds[0])
        print(f"Gambar: {fname} | Prediksi MES: {pred_class} | Confidence: {confidence:.2f}")
        
        # Buat Heatmap
        heatmap = make_gradcam_heatmap(inputs, model, last_conv_name)
        
        # Simpan
        out_name = f"GradCAM_PredMES{pred_class}_{fname}"
        out_path = os.path.join(args.out_dir, out_name)
        save_and_display_gradcam(img_path, heatmap, out_path)

if __name__ == "__main__":
    main()

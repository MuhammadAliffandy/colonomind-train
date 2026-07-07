import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_loader import DataLoader
from features import Extractor
from comparison_models import (
    create_ResNet_50_branch, create_DenseNet_121_branch, 
    create_EfficientNet_B4_branch, create_ConvNeXt_Tiny_branch, 
    create_ViT_B_16_branch, build_hybrid_comparison_model
)

# Configuration
COMMON_EPOCHS = 30
COMMON_LR = 1e-4
BATCH_SIZE = 16

def focal_loss(gamma=2.5, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss

def train_model(model_name, branch_builder, train_data, val_data, class_weight_dict):
    print(f"\\n🚀 Training {model_name} Hybrid Pipeline...")
    model = build_hybrid_comparison_model(branch_builder)
    
    model.compile(
        optimizer=Adam(learning_rate=COMMON_LR),
        loss=focal_loss(),
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, mode='max')
    ]

    model.fit(
        train_data[0], train_data[1],
        validation_data=(val_data[0], val_data[1]),
        batch_size=BATCH_SIZE,
        epochs=COMMON_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    save_dir = f"../results/{model_name}_Experiment"
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"{model_name}_hybrid.h5"))
    print(f"✅ Saved {model_name} weights.")

if __name__ == "__main__":
    print("Loading data...")
    # Assume data is loaded correctly via standard pipeline
    # train_inputs = [X_img_train_bal, X_feat_train_bal, X_train_umap]
    # y_train = y_train_cat_bal
    print("Starting Comparison Model Training Setup...")
    # train_model("ResNet-50", create_ResNet_50_branch, ...)
    # train_model("ViT-B-16", create_ViT_B_16_branch, ...)
    print("Comparison pipeline configured.")

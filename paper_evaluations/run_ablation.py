import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sys
sys.path.append('..')
from src.data_loader import load_dataset
from src.model import build_hybrid_model, create_SE2CNN_model
from src.train import focal_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

def get_ablation_model(scenario, input_shape, feat_shape, num_classes):
    """
    Returns a modified model based on the ablation scenario requested.
    Scenarios:
    1: CNN Only
    2: Handcrafted Features Only
    3: CNN + Handcrafted Features (No UMAP/Tree)
    4: UMAP Only
    5: Handcrafted + UMAP
    6: Full Hybrid Model
    """
    from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    
    img_in = Input(shape=input_shape)
    feat_in = Input(shape=feat_shape)
    umap_in = Input(shape=(2,))
    
    cnn_base = create_SE2CNN_model(input_shape, num_classes, 0.4)
    x_cnn = Dense(64, activation='relu')(cnn_base(img_in))
    
    x_feat = Dense(64, activation='relu')(feat_in)
    x_umap = Dense(32, activation='relu')(umap_in)
    
    if scenario == 1:
        x = x_cnn
        inputs = img_in
    elif scenario == 2:
        x = x_feat
        inputs = feat_in
    elif scenario == 3:
        x = Concatenate()([x_cnn, x_feat])
        inputs = [img_in, feat_in]
    elif scenario == 4:
        x = x_umap
        inputs = umap_in
    elif scenario == 5:
        x = Concatenate()([x_feat, x_umap])
        inputs = [feat_in, umap_in]
    else: # Scenario 6 (Full)
        x = Concatenate()([x_cnn, x_feat, x_umap])
        inputs = [img_in, feat_in, umap_in]
        
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, choices=[1,2,3,4,5,6], default=6, help="Ablation Scenario 1-6")
    parser.add_argument('--domain', type=str, choices=['intra', 'cross', 'multi'], default='intra')
    args = parser.parse_args()
    
    print(f"=== Running Ablation Study ===")
    print(f"Scenario: {args.scenario}")
    print(f"Domain: {args.domain}")
    print("===============================\n")
    
    # Note: In a real run, this script would load the dataset using `load_dataset`, 
    # extract the specific inputs required by the scenario, and run `model.fit()`.
    # It loops through all scenarios and generates the Ablation Table (Top Section Major 5).
    
    print("Script skeleton ready. Ensure to pass appropriate dataset paths to evaluate ablation accuracy!")

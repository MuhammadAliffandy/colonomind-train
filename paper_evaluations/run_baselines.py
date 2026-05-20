import argparse
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
# ConvNeXt and ViT can be loaded from keras_cv or transformers depending on setup.
# Here we use the TF Keras built-in applications for baseline CNNs.

def get_baseline_model(model_name, input_shape, num_classes):
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    
    if model_name == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'densenet':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Model {model_name} not supported natively in this skeleton.")
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet', 'densenet', 'efficientnet', 'convnext', 'vit'], required=True)
    args = parser.parse_args()
    
    print(f"=== Baseline Model Training ===")
    print(f"Model: {args.model}")
    print("===============================\n")
    
    # In practice:
    # 1. Load data via src.data_loader
    # 2. Extract X_img_train, y_train_cat
    # 3. Compile and train baseline model
    # 4. Compare validation accuracy against Mod-SE(2) Hybrid model

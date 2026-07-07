import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB4, ConvNeXtTiny
from transformers import TFViTModel

def create_ResNet_50_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers: layer.trainable = False
    for layer in base_model.layers[-15:]: 
        if not isinstance(layer, BatchNormalization): layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="ResNet_Branch")

def create_DenseNet_121_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers: layer.trainable = False
    for layer in base_model.layers[-15:]:
        if not isinstance(layer, BatchNormalization): layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="DenseNet_Branch")

def create_EfficientNet_B4_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers: layer.trainable = False
    for layer in base_model.layers[-15:]:
        if not isinstance(layer, BatchNormalization): layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="EfficientNet_Branch")

def create_ConvNeXt_Tiny_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    base_model = ConvNeXtTiny(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers: layer.trainable = False
    for layer in base_model.layers[-15:]:
        if not isinstance(layer, BatchNormalization): layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="ConvNeXt_Branch")

def create_ViT_B_16_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_vit')
    vit_model = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model.trainable = False 
    outputs = vit_model(pixel_values=image_input)
    cls_token = Lambda(lambda x: x[:, 0, :])(outputs.last_hidden_state)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(cls_token)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="ViT_Branch")

def build_hybrid_comparison_model(branch_builder_func, image_input_shape=(224, 224, 3), feat_input_shape=(20,), umap_feat_shape=(2,), num_classes=4, dropout_rate=0.5):
    # Branch 1: CNN/ViT Architecture
    image_input = Input(shape=image_input_shape, name='image_input')
    cnn_branch = branch_builder_func(image_input_shape, dropout_rate)
    x_cnn = cnn_branch(image_input)
    x_cnn = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Dropout(dropout_rate)(x_cnn)

    # Branch 2: Handcrafted Feature (Texture)
    feat_input = Input(shape=feat_input_shape, name='feat_input')
    x_feat = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(feat_input)
    x_feat = BatchNormalization()(x_feat)
    x_feat = Dropout(dropout_rate)(x_feat)

    # Branch 3: UMAP Feature
    umap_input = Input(shape=umap_feat_shape, name='umap_input')
    x_umap = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(umap_input)
    x_umap = BatchNormalization()(x_umap)
    x_umap = Dropout(dropout_rate)(x_umap)

    # Fusion
    combined = Concatenate()([x_cnn, x_feat, x_umap])
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
    x = Dropout(dropout_rate)(x)
    
    output = Dense(num_classes, activation='softmax', name='hybrid_output')(x)
    return Model(inputs=[image_input, feat_input, umap_input], outputs=output)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Concatenate, Layer, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB4, ConvNeXtTiny
import tensorflow_hub as hub

def focal_loss(gamma=2.5, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss

def build_hybrid_model(branch_builder_func, image_input_shape, feat_input_shape, umap_feat_shape, num_classes, dropout_rate=0.5):
    image_input = Input(shape=image_input_shape, name='image_input')
    cnn_branch = branch_builder_func(image_input_shape, dropout_rate)
    x_cnn = cnn_branch(image_input)
    # Increased CNN capacity to 128 as per reviewer request
    x_cnn = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Dropout(dropout_rate)(x_cnn)

    feat_input = Input(shape=feat_input_shape, name='feat_input')
    x_feat = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(feat_input)
    x_feat = BatchNormalization()(x_feat)
    x_feat = Dropout(dropout_rate)(x_feat)

    umap_input = Input(shape=umap_feat_shape, name='umap_input')
    x_umap = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(umap_input)
    x_umap = BatchNormalization()(x_umap)
    x_umap = Dropout(dropout_rate)(x_umap)

    combined = Concatenate()([x_cnn, x_feat, x_umap])
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
    x = Dropout(dropout_rate)(x)
    
    output = Dense(num_classes, activation='softmax', name='hybrid_output')(x)
    model = Model(inputs=[image_input, feat_input, umap_input], outputs=output)
    return model

def create_ResNet_50_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    # Preprocess input specifically for ResNet50 (Caffe style)
    x = Lambda(tf.keras.applications.resnet50.preprocess_input)(image_input)
    aug = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    aug = tf.keras.layers.RandomRotation(0.2)(aug)
    aug = tf.keras.layers.RandomZoom(0.2)(aug)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=aug)
    # Fully freeze backbone
    for layer in base_model.layers: layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="ResNet_Branch")

def create_DenseNet_121_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    # Preprocess input specifically for DenseNet121
    x = Lambda(tf.keras.applications.densenet.preprocess_input)(image_input)
    aug = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    aug = tf.keras.layers.RandomRotation(0.2)(aug)
    aug = tf.keras.layers.RandomZoom(0.2)(aug)
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=aug)
    # Fully freeze backbone
    for layer in base_model.layers: layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="DenseNet_Branch")

def create_EfficientNet_B4_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    # Preprocess input specifically for EfficientNetB4
    x = Lambda(tf.keras.applications.efficientnet.preprocess_input)(image_input)
    aug = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    aug = tf.keras.layers.RandomRotation(0.2)(aug)
    aug = tf.keras.layers.RandomZoom(0.2)(aug)
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_tensor=aug)
    # Fully freeze backbone
    for layer in base_model.layers: layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="EfficientNet_Branch")

def create_ConvNeXt_Tiny_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_cnn')
    # Preprocess input specifically for ConvNeXtTiny
    x = Lambda(tf.keras.applications.convnext.preprocess_input)(image_input)
    aug = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    aug = tf.keras.layers.RandomRotation(0.2)(aug)
    aug = tf.keras.layers.RandomZoom(0.2)(aug)
    base_model = ConvNeXtTiny(weights='imagenet', include_top=False, input_tensor=aug)
    # Fully freeze backbone
    for layer in base_model.layers: layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="ConvNeXt_Branch")

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

def create_ViT_B_16_branch(input_shape, dropout_rate=0.5):
    image_input = Input(shape=input_shape, name='image_input_vit')
    # Preprocess for ViT-B16 TF-Hub (expects [-1, 1])
    x = Lambda(lambda img: (img / 127.5) - 1.0)(image_input)
    aug = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    aug = tf.keras.layers.RandomRotation(0.2)(aug)
    aug = tf.keras.layers.RandomZoom(0.2)(aug)
    # Backbone is inherently frozen in the wrapper
    x = ViT_B16_Wrapper()(aug)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inputs=image_input, outputs=x, name="ViT_Branch")

MODEL_BUILDERS = {
    'ResNet-50': create_ResNet_50_branch,
    'DenseNet-121': create_DenseNet_121_branch,
    'EfficientNet-B4': create_EfficientNet_B4_branch,
    'ConvNeXt-Tiny': create_ConvNeXt_Tiny_branch,
    'ViT-B-16': create_ViT_B_16_branch
}

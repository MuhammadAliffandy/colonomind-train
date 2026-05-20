import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# === SE(2) CNN HELPER FUNCTIONS ===

def ToLinearIndex(ij, NiNj):
    return ij[0] * NiNj[0] + ij[1]

def CoordRotationInv(ij, NiNj, theta):
    centeri = np.floor(NiNj[0] / 2)
    centerj = np.floor(NiNj[1] / 2)
    ijOld = np.zeros([2])
    ijOld[0] = np.cos(theta) * (ij[0] - centeri) + np.sin(theta) * (ij[1] - centerj) + centeri
    ijOld[1] = -np.sin(theta) * (ij[0] - centeri) + np.cos(theta) * (ij[1] - centerj) + centerj
    return ijOld

def LinIntIndicesAndWeights(ij, NiNj):
    i, j = ij
    Ni, Nj = NiNj
    i1 = int(np.floor(i))
    i2 = i1 + 1
    j1 = int(np.floor(j))
    j2 = j1 + 1
    ti = i - i1
    tj = j - j1
    w11 = (1 - ti) * (1 - tj)
    w12 = (1 - ti) * tj
    w21 = ti * (1 - tj)
    w22 = ti * tj
    indicesAndWeights = []
    if (0 <= i1 < Ni) and (0 <= j1 < Nj):
        indicesAndWeights.append([i1, j1, w11])
    if (0 <= i1 < Ni) and (0 <= j2 < Nj):
        indicesAndWeights.append([i1, j2, w12])
    if (0 <= i2 < Ni) and (0 <= j1 < Nj):
        indicesAndWeights.append([i2, j1, w21])
    if (0 <= i2 < Ni) and (0 <= j2 < Nj):
        indicesAndWeights.append([i2, j2, w22])
    return indicesAndWeights

def RotationOperatorMatrixSparse(NiNj, theta, diskMask=True, linIndOffset=0):
    Ni, Nj = NiNj
    cij = np.floor(Ni / 2)
    idx, vals = [], []
    for i in range(NiNj[0]):
        for j in range(NiNj[0]):
            if not(diskMask) or ((i - cij) * (i - cij) + (j - cij) * (j - cij) <= (cij + 0.5) * (cij + 0.5)):
                linij = ToLinearIndex([i, j], NiNj)
                ijOld = CoordRotationInv([i, j], NiNj, theta)
                linIntIndicesAndWeights = LinIntIndicesAndWeights(ijOld, NiNj)
                for indexAndWeight in linIntIndicesAndWeights:
                    indexOld = [indexAndWeight[0], indexAndWeight[1]]
                    linIndexOld = ToLinearIndex(indexOld, NiNj)
                    weight = indexAndWeight[2]
                    idx.append((linij + linIndOffset, linIndexOld))
                    vals.append(weight)
    return tuple(idx), tuple(vals)

def MultiRotationOperatorMatrixSparse(NiNj, Ntheta, periodicity=2 * np.pi, diskMask=True):
    idx, vals = [], []
    for r in range(Ntheta):
        idxr, valsr = RotationOperatorMatrixSparse(NiNj, periodicity * r / Ntheta, linIndOffset=r * NiNj[0] * NiNj[1], diskMask=diskMask)
        idx += idxr
        vals += valsr
    return idx, vals

# === MOD-SE(2) CNN LAYERS ===

def GroupConv2D(filters, kernel_size, strides=(1, 1), padding='same', groups=3):
    def layer(x):
        # Native Keras Group Convolution — highly optimized and avoids graph splitting issues on GPU
        x_out = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=kernel_size,
            strides=strides, 
            padding=padding,
            groups=groups
        )(x)
        x_out = BatchNormalization()(x_out)
        x_out = tf.keras.layers.Activation('relu')(x_out)
        return x_out
    return layer

def SE2MaxPooling2D(pool_size=(2, 2)):
    def layer(x):
        return tf.keras.layers.MaxPooling2D(pool_size=pool_size)(x)
    return layer

def SE2LiftingLayer(x):
    # Use dynamic shape to avoid issues with None batch dim on GPU
    shape = tf.shape(x)
    static_shape = x.shape.as_list()
    H, W, C = static_shape[1], static_shape[2], static_shape[3]
    assert C is not None and C % 3 == 0, "Number of input channels must be divisible by 3"
    group_size = C // 3
    x = tf.keras.layers.Reshape((H, W, 3, group_size))(x)
    x = tf.keras.layers.Permute((1, 2, 4, 3))(x)
    return x

def create_SE2CNN_model(input_shape, num_classes, dropout_rate=0.5):
    input_layer = Input(shape=input_shape)
    x = input_layer
    x = GroupConv2D(32, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = GroupConv2D(64, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = GroupConv2D(128, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = GroupConv2D(256, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = GroupConv2D(512, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = GroupConv2D(1024, (3, 3))(x)
    x = SE2MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = SE2LiftingLayer(x)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(1056, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

def build_hybrid_model(image_input_shape, feat_input_shape, umap_feat_shape, num_classes, dropout_rate=0.4):
    # CNN Branch
    image_input_se2 = Input(shape=image_input_shape, name='image_input_se2')
    cnn_branch = create_SE2CNN_model(image_input_shape, num_classes, dropout_rate)
    x_se2 = cnn_branch(image_input_se2)
    x_se2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x_se2)
    x_se2 = BatchNormalization()(x_se2)
    x_se2 = Dropout(dropout_rate)(x_se2)

    # Handcrafted Feature Branch
    feat_input = Input(shape=feat_input_shape, name='feat_input')
    x_feat = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(feat_input)
    x_feat = BatchNormalization()(x_feat)
    x_feat = Dropout(dropout_rate)(x_feat)

    # UMAP Feature Branch
    umap_input = Input(shape=umap_feat_shape, name='umap_feat_input')
    x_umap = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(umap_input)
    x_umap = BatchNormalization()(x_umap)
    x_umap = Dropout(dropout_rate)(x_umap)

    # Fusion
    combined = Concatenate()([x_se2, x_feat, x_umap])
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[image_input_se2, feat_input, umap_input], outputs=output)
    return model

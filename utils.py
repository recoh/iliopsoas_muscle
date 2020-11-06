import json
from typing import Any

import numpy as np
import scipy.ndimage as nd
from keras.activations import elu
from keras.engine import Input
from keras.layers import Conv3D, add, concatenate, Conv3DTranspose
from keras.models import Model


def selu(x):  # https://gist.github.com/naure/78bc7a881a9db17e366093c81425184f
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha) 
    
    
def downward_layer(input_layer, n_convolutions: int, n_output_channels: int) -> (Any, Any):
    inl = input_layer

    for _ in range(n_convolutions):
        inl = Conv3D(filters=(n_output_channels // 2), kernel_size=5, activation=selu, padding='same', kernel_initializer='he_normal')(inl)
    add_l = add([inl, input_layer])
    downsample = Conv3D(filters=n_output_channels, kernel_size=2, strides=2, activation=selu, padding='same', kernel_initializer='he_normal')(add_l)
    downsample = (downsample)
    return downsample, add_l

    
def upward_layer(input0, input1, n_convolutions: int, n_output_channels: int) -> Any:
    merged = concatenate([input0, input1], axis=4)
    inl = merged

    for _ in range(n_convolutions):
        inl = Conv3D((n_output_channels * 4), kernel_size=5, padding='same', activation=selu, kernel_initializer='he_normal')(inl)
    add_l = add([inl, merged])
    shape = add_l.get_shape().as_list()
    new_shape = (1, shape[1] * 2, shape[2] * 2, shape[3] * 2, n_output_channels)
    upsample = Conv3DTranspose(n_output_channels, (2, 2, 2), strides=(2, 2, 2), activation=selu, padding='same', kernel_initializer='he_normal')(add_l)
    return (upsample)


def iliopsoas_seg_net(input_size):

    # Down 1
    inputs = Input(input_size)
    conv_1 = Conv3D(8, kernel_size=5, strides=1, activation=selu, padding='same', kernel_initializer='he_normal')(inputs)
    repeat_1 = concatenate(8 * [inputs], axis=-1)
    add_1 = add([conv_1, repeat_1])
    down_1 = Conv3D(16, 2, strides=2, activation=selu, padding='same', kernel_initializer='he_normal')(add_1)

    # Down 2,3,4
    down_2, add_2 = downward_layer(down_1, 2, 32)
    down_3, add_3 = downward_layer(down_2, 3, 64)
    down_4, add_4 = downward_layer(down_3, 3, 128)

    # Bottom
    conv_5_1 = Conv3D(128, kernel_size=(5, 5, 5), activation=selu, padding='same', kernel_initializer='he_normal')(down_4)
    conv_5_2 = Conv3D(128, kernel_size=(5, 5, 5), activation=selu, padding='same', kernel_initializer='he_normal')(conv_5_1)
    conv_5_3 = Conv3D(128, kernel_size=(5, 5, 5), activation=selu, padding='same', kernel_initializer='he_normal')(conv_5_2)
    add5 = add([conv_5_3, down_4])
    aux_shape = add5.get_shape()
    upsample_5 = Conv3DTranspose(64, (2, 2, 2), padding='same', activation=selu, strides=(2, 2, 2), kernel_initializer='he_normal')(add5)

    # Up 6,7,8
    upsample_6 = upward_layer(upsample_5, add_4, 3, 32)
    upsample_7 = upward_layer(upsample_6, add_3, 3, 16)
    upsample_8 = upward_layer(upsample_7, add_2, 2, 8)

    # Up 9
    merged_9 = concatenate([upsample_8, add_1], axis=4)
    conv_9_1 = Conv3D(16, kernel_size=(5, 5, 5), activation=selu, padding='same', kernel_initializer='he_normal')(merged_9)

    add_9 = add([conv_9_1, merged_9])
    conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), activation=selu, padding='same', kernel_initializer='he_normal')(add_9)

    sigmoid = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal', activation='sigmoid')(conv_9_2)

    model = Model(inputs=inputs, outputs=sigmoid)
    return model


BONE_LABELS = {
    'shoulder': {'right': 1, 'left': 2},
    'femur_head': {'right': 11, 'left': 12},
    'femur_base': {'right': 21, 'left': 22},
}

BONE_LABEL_ALIASES = {
    'hip': 'femur_head',
    'knee': 'femur_base',
}


def load_landmarks() -> dict:
    with open('landmarks/bone_joints.json', 'r') as json_file:
        landmarks_dict = json.load(json_file)

    output = dict()
    for bone_type in BONE_LABELS:
        output[bone_type] = dict()
        bone_type_z = list()
        for bone in BONE_LABELS[bone_type]:
            bone_index = BONE_LABELS[bone_type][bone]
            bone_str = str(bone_index)
            if bone_str not in landmarks_dict.keys():
                print('Cannot find {}, {} in detected bone data'.format(bone_type, bone))
                output[bone_type][bone] = np.array([np.NAN, np.NAN, np.NAN])
            else:
                output[bone_type][bone] = np.array(landmarks_dict[bone_str], dtype='int')
                bone_type_z.append(landmarks_dict[bone_str][2])
        if np.any(np.isfinite(bone_type_z)):
            output[bone_type]['z'] = int(np.nanmean(bone_type_z))
        else:
            output[bone_type]['z'] = None

    for alias_key in BONE_LABEL_ALIASES:
        output[alias_key] = output[BONE_LABEL_ALIASES[alias_key]]

    return output


def largest_connected_components(img: np.ndarray, n_components: int = 1, structure: np.ndarray = None) -> np.ndarray:
    labels, nb_labels = nd.label(img > 0, structure=structure)
    res = np.zeros(img.shape, dtype='int')
    if n_components > nb_labels:
        print('requested number of components ({}) larger than image contains ({})'.format(n_components, nb_labels))
        n_components = nb_labels
    hist = nd.histogram(labels, 1, nb_labels, nb_labels)
    i_sort = np.argsort(hist)[::-1][:n_components] + 1
    for i_n, i in enumerate(i_sort, start=1):
        res[labels == i] = i_n

    return res


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype('float32') / np.percentile(img, 99)
    img[img > 1.0] = 0.975
    return img

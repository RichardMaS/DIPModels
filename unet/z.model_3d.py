from keras import backend as K
from keras.applications import imagenet_utils
from keras.engine.topology import get_source_inputs
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv3D
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalAveragePooling3D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalMaxPooling3D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling3D
from keras.layers import UpSampling2D
from keras.layers import UpSampling3D
from keras.models import Model

## Merge with model
def conv_block(x, filter_size, dropout_rate, use_batch_norm, name):
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    
    block = Conv3D(filter_size, kernel_size=(3, 3, 3), padding='same', name=name + '_1_conv')(x)
    if use_batch_norm:
        block = BatchNormalization(axis=bn_axis, name=name + '_1_bn')(block)
    block = Activation('relu')(block)
    block = Conv3D(filter_size*2, kernel_size=(3, 3, 3), padding='same', name=name + '_2_conv')(block)
    if use_batch_norm:
        block = BatchNormalization(axis=bn_axis, name=name + '_2_bn')(block)
    block = Activation('relu')(block)
    if dropout_rate > 0:
        block = Dropout(dropout_rate, name=name + '_1_dp')(block)
    return block

def unet_3d(filter_size = 32, weights=None, input_tensor=None, input_shape=None, pooling=None, dropout_rate=0.2, use_batch_norm=True):
    include_top = False
    nb_masks_classes = 1
    
    input_shape = imagenet_utils._obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=221,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    
    conv_1 = conv_block(img_input, 1*filter_size, 0, use_batch_norm, name='conv_1')
    pool_1 = MaxPooling3D(pool_size=(2, 2, 2), name='conv_1_pool')(conv_1)

    conv_2 = conv_block(pool_1, 2*filter_size, 0, use_batch_norm, name='conv_2')
    pool_2 = MaxPooling3D(pool_size=(2, 2, 2), name='conv_2_pool')(conv_2)

    conv_3 = conv_block(pool_2, 4*filter_size, 0, use_batch_norm, name='conv_3')
    pool_3 = MaxPooling3D(pool_size=(2, 2, 2), name='conv_3_pool')(conv_3)

    conv_4 = conv_block(pool_3, 8*filter_size, 0, use_batch_norm, name='conv_4')
    pool_4 = MaxPooling3D(pool_size=(2, 2, 2), name='conv_4_pool')(conv_4)
    
    conv_5 = conv_block(pool_4, 16*filter_size, 0, use_batch_norm, name='conv_5')
    pool_5 = MaxPooling3D(pool_size=(2, 2, 2), name='conv_5_pool')(conv_5)

    conv_6 = conv_block(pool_5, 32*filter_size, 0, use_batch_norm, name='conv_6')

    up_5 = Concatenate(axis=bn_axis, name='up_5_concat')([UpSampling3D(size=(2, 2, 2))(conv_6), conv_5])
    up_conv_5 = conv_block(up_5, 16*filter_size, 0, use_batch_norm, name='up_5')

    up_4 = Concatenate(axis=bn_axis, name='up_4_concat')([UpSampling3D(size=(2, 2, 2))(up_conv_5), conv_4])
    up_conv_4 = conv_block(up_4, 8*filter_size, 0, use_batch_norm, name='up_4')

    up_3 = Concatenate(axis=bn_axis, name='up_3_concat')([UpSampling3D(size=(2, 2, 2))(up_conv_4), conv_3])
    up_conv_3 = conv_block(up_3, 4*filter_size, 0, use_batch_norm, name='up_3')

    up_2 = Concatenate(axis=bn_axis, name='up_2_concat')([UpSampling3D(size=(2, 2, 2))(up_conv_3), conv_2])
    up_conv_2 = conv_block(up_2, 2*filter_size, 0, use_batch_norm, name='up_2')

    up_1 = Concatenate(axis=bn_axis, name='up_1_concat')([UpSampling3D(size=(2, 2, 2))(up_conv_2), conv_1])
    up_conv_1 = conv_block(up_1, 1*filter_size, dropout_rate, use_batch_norm, name='up_1')

    conv_final = Conv3D(nb_masks_classes, (1, 1, 1), activation='sigmoid', name='output')(up_conv_1)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    model = Model(inputs=inputs, outputs=conv_final, name='UNet_' + str(filter_size))
    return model

def preprocess_input(x, data_format=None):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format, mode='torch')

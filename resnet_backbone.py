from keras import backend as K
from keras.layers import Activation
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import ZeroPadding2D
from keras.models import Model

CHANNEL_AXIS = -1 if K.image_data_format() == 'channels_last' else 1

def conv_block_double(x, filter_size, dropout_rate, use_batch_norm, name):
    x = conv_bn_relu_block(filters=filter_size, kernel_size=(3, 3), 
                           use_batch_norm=use_batch_norm, 
                           conv_name=name + '_1_conv', 
                           bn_name=name + '_1_bn',
                           relu_name=name + '_1_activation')(x)
    x = conv_bn_relu_block(filters=filter_size, kernel_size=(3, 3), 
                           use_batch_norm=use_batch_norm, 
                           conv_name=name + '_2_conv', 
                           bn_name=name + '_2_bn',
                           relu_name=name + '_2_activation')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name=name + '_1_dp')(s)
    return x
    
# https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/resnet.py
def conv_bn_relu_block(**kwargs):
    """ Helper to build a conv -> BN -> relu residual unit activation function.
        This is the original ResNet v1 scheme in https://arxiv.org/abs/1512.03385
    """
    filters = kwargs["filters"]
    kernel_size = kwargs["kernel_size"]
    strides = kwargs.setdefault("strides", (1, 1))
    dilation_rate = kwargs.setdefault("dilation_rate", (1, 1))
    kernel_initializer = kwargs.setdefault("kernel_initializer", "he_normal")
    padding = kwargs.setdefault("padding", "same")
    kernel_regularizer = kwargs.setdefault("kernel_regularizer", l2(1.e-4))
    use_batch_norm = kwargs.setdefault("use_batch_norm", True)
    conv_name = kwargs.setdefault("conv_name", None)
    bn_name = kwargs.setdefault("bn_name", None)
    relu_name = kwargs.setdefault("relu_name", None)

    def f(x):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        if use_batch_norm:
            x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
        x = Activation("relu", name=relu_name)(x)
        return x

    return f


def bn_relu_conv_block(**conv_params):
    """ Helper to build a BN -> relu -> conv residual unit with full pre-activation function.
        This is the ResNet v2 scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    use_batch_norm = kwargs.setdefault("use_batch_norm", True)
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)

    def f(x):
        if use_batch_norm:
            x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
        x = Activation("relu", name=relu_name)(x)
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        return x

    return f


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
        shape: (None, h, w, d) => (None, h, w, filters[-1])
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    ## input_tensor: (None, n, n, filters0)
    # res2x_branch2a: (None, n, n, filters1) kernel_size(1)*filters0*filter1 + filters1
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    # bn2x_branch2a: (None, n, n, filters1) filters1*4
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    # activation_...: (None, n, n, filters1) 0
    x = Activation('relu')(x)

    # res2x_branch2b: (None, n, n, filters2) kernel_size(9)*filters1*filters2 + filters2
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # bn2x_branch2b: (None, n, n, filters2) filters2*4
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # activation_...: (None, n, n, filters2) 0
    x = Activation('relu')(x)

    # res2x_branch2c: (None, n, n, filters2) kernel_size(1)*filters2*filters3 + filters3
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # bn2b_branch2c: (None, n, n, filters3) filters3*4
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
    # add_k (Add): (None, n, n, filters3) 0
    x = Add()([x, input_tensor])
    # res2x_out (Activation): (None, n, n, filters3) 0
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block. 
        h1, w1 = ([h, w]−kernel_size+2*padding=0)/strides+1 = ceiling([h, w]/strides)
        shape: (None, h, w, d) => (None, h1, w1, filters[-1])

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    ## Dimension reduction: x=(None, h, w, filters0), 
    ## n = (h/w−kernel_size+2*padding=0)/strides+1 = ceiling(n/strides)
    # res2x_branch2a: (None, n, n, filters1) kernel_size(1)*filters0*filter1 + filters1
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    # bn2x_branch2a: (None, n, n, filters1) filters1*4
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    # activation_...: (None, n, n, filters1) 0
    x = Activation('relu')(x) # (None, h, w, filters1)
    
    # res2x_branch2b: (None, n, n, filters2) kernel_size(9)*filters1*filters2 + filters2
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    # bn2x_branch2b: (None, n, n, filters2) filters2*4
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # activation_...: (None, n, n, filters2) 0
    x = Activation('relu')(x)
    
    # res2x_branch2c: (None, n, n, filters3) kernel_size(1)*filters2*filters3 + filters3
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # bn2x_branch2c: (None, n, n, filters3) filters3*4
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
    # res2x_branch1: (None, n, n, filters3) kernel_size(1)*filters0*filters3 + filters3
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    # bn2x_branch1: (None, n, n, filters3) filters3*4
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    
    # add_k (Add): (None, n, n, filters3) 0
    x = Add()([x, shortcut])
    # res2x_out (Activation): (None, n, n, filters3) 0
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    
    return x


def resnet_graph_v1(input_image, architecture):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layres
    """
    assert architecture in ["resnet50", "resnet101"]
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    
    # Stage 1
    # input_image: (None, 1024, 1024, 3) 0
    # x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_image) # conv1_pad: (None, 1030, 1030, 3) 0
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(x) # conv1: (None, 512, 512, 64) 49*3*64+64
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x) # bn_conv1: (None, 512, 512, 64) 64*4
    C1 = x = Activation('relu')(x) # activation_1: (None, 512, 512, 64) 0
    
    # Stage 2
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x) # max_pooling2d_1: (None, 256, 256, 64) 0
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1)) # res2a_out: (None, 256, 256, 256), s=1
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b') # res2b_out: (None, 256, 256, 256)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c') # res2c_out: (None, 256, 256, 256)
    
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a') # res3a_out: (None, 128, 128, 512), s=2
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b') # res3b_out: (None, 128, 128, 512)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c') # res3c_out: (None, 128, 128, 512)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d') # res3d_out: (None, 128, 128, 512)
    
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a') # res3a_out: (None, 64, 64, 1024), s=2
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i)) # res3x_out: (None, 64, 64, 1024)
    C4 = x
    
    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    C5 = x
    
    return [C1, C2, C3, C4, C5]


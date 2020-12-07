from keras import backend as K
from keras.applications import imagenet_utils
from keras.engine.topology import get_source_inputs
from keras.layers import Activation
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.utils import Sequence
from keras.utils import to_categorical
import keras.optimizers
import keras.callbacks

import numpy as np
import tensorflow as tf
import os
import skimage
import time
import random
import multiprocessing

from DIPModels.utils_g import utils_image
from DIPModels.utils_g import utils_keras
from DIPModels.utils_g import utils_misc
from . import utils as unet_utils
from skimage import io


CHANNEL_AXIS = -1 if K.image_data_format() == 'channels_last' else 1

def conv_bn_relu_block(**kwargs):
    """ Build a conv -> BN (maybe) -> relu block. """
    filters = kwargs['filters']
    kernel_size = kwargs['kernel_size']
    strides = kwargs.setdefault('strides', (1, 1))
    dilation_rate = kwargs.setdefault('dilation_rate', (1, 1))
    kernel_initializer = kwargs.setdefault('kernel_initializer', 'he_normal')
    padding = kwargs.setdefault('padding', 'same')
    kernel_regularizer = kwargs.setdefault('kernel_regularizer', None)
    use_batch_norm = kwargs.setdefault('use_batch_norm', True)
    conv_name = kwargs.setdefault('conv_name', None)
    bn_name = kwargs.setdefault('bn_name', None)
    relu_name = kwargs.setdefault('relu_name', None)

    def f(x):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   strides=strides, padding=padding,
                   dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name=conv_name)(x)
        if use_batch_norm:
            x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x

    return f

def unet_conv_block(x, filters, use_batch_norm, dropout_rate, name):
    """ Basic UNet convolution block: 2 conv->bn->relu + dropout(maybe). """
    x = conv_bn_relu_block(filters=filters, kernel_size=(3, 3), 
                           use_batch_norm=use_batch_norm, 
                           conv_name=name + '_1_conv', 
                           bn_name=name + '_1_bn',
                           relu_name=name + '_1_activation')(x)
    x = conv_bn_relu_block(filters=filters, kernel_size=(3, 3), 
                           use_batch_norm=use_batch_norm, 
                           conv_name=name + '_2_conv', 
                           bn_name=name + '_2_bn',
                           relu_name=name + '_2_activation')(x)
    
    ## Add a dropout layer
    if dropout_rate:
        x = Dropout(dropout_rate, name=name + '_1_dp')(x)
    return x


def unet_encoder(x, config):
    """ Basic UNet encoder. """
    filter_size = config.FILTER_SIZE
    use_batch_norm = config.ENCODER_USE_BN
    dropout_rate = config.DROPOUT_RATE
    
    # Model Input Size must be dividable by 2**NUM_LAYERS
    h, w = K.int_shape(x)[1:-1]
    if h % 2**len(filter_size) or w % 2**len(filter_size):
        raise Exception("Image size must be dividable by " + 
                        str(2**len(filter_size)) +
                        "to avoid fractions when downscaling and upscaling.")
    
    encoder_layers = []
    for i, filters in enumerate(filter_size[:-1]):
        name = 'encoder_' + str(i + 1)
        x = unet_conv_block(x, filters, use_batch_norm, dropout_rate, name)
        encoder_layers.append(x)
        x = MaxPooling2D(pool_size=(2, 2), name=name + '_pool')(x)
    
    x = unet_conv_block(x, filter_size[-1], use_batch_norm, dropout_rate,
                        name='encoder_' + str(len(filter_size)))
    
    return encoder_layers + [x]


def unet_decoder(encoder_layers, config):
    """ Basic UNet decoder. """
    upsampling_mode = config.UPSAMPLING_MODE
    use_batch_norm = config.DECODER_USE_BN
    dropout_rate = config.DROPOUT_RATE
    assert upsampling_mode in ['nearest', 'Conv2DTranspose'], "Unsupported upsampling mode!"
    
    x = encoder_layers[-1]
    for i in range(len(encoder_layers)-2, -1, -1):
        name = 'decoder_' + str(i+1)
        filter_size = K.int_shape(encoder_layers[i])[-1]
        if upsampling_mode == 'nearest':
            x = UpSampling2D(size=(2, 2), name=name + '_upsampling')(x)
            x = Conv2D(filter_size, kernel_size=(2, 2), padding='same', name=name + '_upconv')(x)
        else:
            x = Conv2DTranspose(filter_size, kernel_size=(2, 2), strides=(2, 2), padding='same', name=name + '_upconv')(x)
        if use_batch_norm:
            x = BatchNormalization(axis=CHANNEL_AXIS, name=name + '_bn')(x)
        x = Activation('relu', name=name + '_activation')(x)
        
        x = Concatenate(axis=CHANNEL_AXIS, name=name + '_concat')([x, encoder_layers[i]])
        x = unet_conv_block(x, filter_size, use_batch_norm, dropout_rate, name=name)
        
    return x

## only used for view the structure. Function not called.
def unet_2d_vanilla(x, filter_size, use_batch_norm=False, dropout_rate=0.0):
    conv_1 = unet_conv_block(x, 1*filter_size, dropout_rate, use_batch_norm, name='conv_1')
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='conv_1_pool')(conv_1)

    conv_2 = unet_conv_block(pool_1, 2*filter_size, dropout_rate, use_batch_norm, name='conv_2')
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='conv_2_pool')(conv_2)

    conv_3 = unet_conv_block(pool_2, 4*filter_size, dropout_rate, use_batch_norm, name='conv_3')
    pool_3 = MaxPooling2D(pool_size=(2, 2), name='conv_3_pool')(conv_3)

    conv_4 = unet_conv_block(pool_3, 8*filter_size, dropout_rate, use_batch_norm, name='conv_4')
    pool_4 = MaxPooling2D(pool_size=(2, 2), name='conv_4_pool')(conv_4)

    conv_5 = unet_conv_block(pool_4, 16*filter_size, dropout_rate, use_batch_norm, name='conv_5')
    pool_5 = MaxPooling2D(pool_size=(2, 2), name='conv_5_pool')(conv_5)

    conv_6 = unet_conv_block(pool_5, 32*filter_size, dropout_rate, use_batch_norm, name='conv_6')

    up_5 = Conv2D(16*filter_size, 2, activation='relu', padding='same', name='up_5_upconv')(UpSampling2D(size=(2, 2))(conv_6))
    up_5 = Concatenate(axis=CHANNEL_AXIS, name='up_5_concat')([up_5, conv_5])
    up_conv_5 = unet_conv_block(up_5, 16*filter_size, dropout_rate, use_batch_norm, name='up_5')

    up_4 = Conv2D(8*filter_size, 2, activation='relu', padding='same', name='up_4_upconv')(UpSampling2D(size=(2, 2))(up_conv_5))
    up_4 = Concatenate(axis=CHANNEL_AXIS, name='up_4_concat')([up_4, conv_4])
    up_conv_4 = unet_conv_block(up_4, 8*filter_size, dropout_rate, use_batch_norm, name='up_4')

    up_3 = Conv2D(4*filter_size, 2, activation='relu', padding='same', name='up_3_upconv')(UpSampling2D(size=(2, 2))(up_conv_4))
    up_3 = Concatenate(axis=CHANNEL_AXIS, name='up_3_concat')([up_3, conv_3])
    up_conv_3 = unet_conv_block(up_3, 4*filter_size, dropout_rate, use_batch_norm, name='up_3')

    up_2 = Conv2D(2*filter_size, 2, activation='relu', padding='same', name='up_2_upconv')(UpSampling2D(size=(2, 2))(up_conv_3))
    up_2 = Concatenate(axis=CHANNEL_AXIS, name='up_2_concat')([up_2, conv_2])
    up_conv_2 = unet_conv_block(up_2, 2*filter_size, dropout_rate, use_batch_norm, name='up_2')

    up_1 = Conv2D(1*filter_size, 2, activation='relu', padding='same', name='up_1_upconv')(UpSampling2D(size=(2, 2))(up_conv_2))
    up_1 = Concatenate(axis=CHANNEL_AXIS, name='up_1_concat')([up_1, conv_1])
    up_conv_1 = unet_conv_block(up_1, 1*filter_size, dropout_rate, use_batch_norm, name='up_1')
    
    ## return image map without logits and scores
    return up_conv_1

## functions imported from resnet_backbone.py

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
    # res2x_branch2a: (None, n, n, filters1) kernel_size(1)*filters0*filters1 + filters1
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

def resnet_encoder(input_image, config, **kwargs):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert config.ARCHITECTURE in ["resnet50", "resnet101"]
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    
    # Stage 1
    # input_image: (None, 1024, 1024, 3) 0
    # x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_image) # conv1_pad: (None, 1030, 1030, 3) 0
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_image) # conv1: (None, 512, 512, 64) 49*3*64+64
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
    block_count = {"resnet50": 5, "resnet101": 22}[config.ARCHITECTURE]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i)) # res3x_out: (None, 64, 64, 1024)
    C4 = x
    
    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    C5 = x
    
    return [C1, C2, C3, C4, C5]


class UNet(object):
    """ Basic 2D UNet model.
        The actual Keras model is in the keras_model property.
    """
    def __init__(self, config, model_dir, weights_path=None, **kwargs):
        self.config = config
        self.weights_path = weights_path
        self.model_dir = model_dir
        
        if config.ENCODER == 'UNet':
            self.encoder = unet_encoder
        elif config.ENCODER == 'ResNet':
            self.encoder = resnet_encoder
        else:
            raise "Unsupport encoder type!"
        
        if config.DECODER == 'UNet':
            self.decoder = unet_decoder
        else:
            raise "Unsupport decoder type!"
        
        self.model_name = config.NAME + '_' + config.ENCODER + '_' + config.DECODER
        self.build_model(**kwargs)
    
    def build_model(self, **kwargs):
        """Build UNet architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        img_input = Input(shape=self.config.MODEL_INPUT_SIZE + (3,), name='input')
        conv_layers = self.encoder(img_input, self.config, **kwargs)
        conv_final = self.decoder(conv_layers, self.config, **kwargs)
        # keep output same size as original if using ResNet encoder
        if self.config.ENCODER == 'ResNet':
            conv_final = UpSampling2D(size=(2, 2))(conv_final)

        
        ## Get logits and set background to 0
        bg_logits = Conv2D(2, (1, 1), name='bg_logits')(conv_final)
        #if self.nb_classes > 2:
        cl_logits = Conv2D(self.config.NUM_CLASSES-1, (1, 1), name='cl_logits')(conv_final)
         #bg_logits: separate pixels into is or is not bg
         #cl_logits: separate pixels based on class (excluding bg)
        
        model = Model(inputs=[img_input], outputs=[bg_logits, cl_logits], 
                      name='UNet2D_' + str(self.config.FILTER_SIZE))
        
        if self.weights_path is not None:
            # print("Loading weights from: " + weights_path)
            model.load_weights(weights_path)
            print("Loading weights from: " + weights_path)
        
        # TODO: Add multi-GPU support.
        #if config.GPU_COUNT > 1:
        #    from parallel_model import ParallelModel
        #    model = ParallelModel(model, config.GPU_COUNT)
        
        self.keras_model = model
        # resnet encoder: 342 layers (excluding input)
        
    
    def get_optimizer(self, learning_rate, decay=1e-6, momentum=0.9, clipnorm=5.0):
        """ Get optimizer. """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, 
                                         clipnorm=clipnorm, nesterov=True)
        optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                                          epsilon=None, decay=decay, amsgrad=False)
        return optimizer
        
    def get_loss(self):
        """ Get loss function. """
        def loss_function(target, output):
            return K.mean(K.categorical_crossentropy(target, K.softmax(output)))
        return {'bg_logits': loss_function, 'cl_logits': loss_function}
    
    def get_metrics(self):
        def bg_iou_metrics(target, output):
            '''To measure: how accurately model distinguishes between cell and bg'''
            # mask: y_true, prediction: y_pred
            y_true, y_pred = target, K.softmax(output)
            return K.mean(utils_keras.iou_coef(y_true, y_pred))
        
        def cl_iou_metrics(target, output):
            '''To measure: how accurately model classifies cell type'''
            y_true, y_pred = target, K.softmax(output)
            # Set true bg region to 0 for output
            y_pred = tf.where(tf.not_equal(y_true, tf.constant(0., dtype=tf.float32)), 
                              y_pred, y_true)
            return K.mean(utils_keras.iou_coef(y_true, y_pred))
        
        return {'bg_logits': bg_iou_metrics, 'cl_logits': cl_iou_metrics}
    
    def get_callbacks(self, log_dir, checkpoint_dir):
        callbacks_list = [keras.callbacks.ModelCheckpoint(checkpoint_dir, # save model after every epoch
                                                          verbose=0, save_weights_only=True),
                          keras.callbacks.TensorBoard(log_dir, histogram_freq=0, #write_graph: visualize layer graph
                                                      write_graph=False, write_images=False)] #write_images: log intermediate outputs
        return callbacks_list
        
    def train(self, train_dataset, val_dataset, epochs,
              use_border_weights=False, use_class_weights=False, border_weights_sigma=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done already, so this actually determines
                the epochs to train in total rather than in this particular
                call.
        """
        # Data generators
        train_generator = DataSequence(train_dataset, config=self.config, 
                                       batch_size=self.config.BATCH_SIZE,
                                       use_border_weights=use_border_weights,
                                       use_class_weights=use_class_weights,
                                       border_weights_sigma=border_weights_sigma,
                                       shuffle=True)
        val_generator = DataSequence(val_dataset, config=self.config, 
                                     batch_size=self.config.BATCH_SIZE,
                                     use_border_weights=use_border_weights,
                                     use_class_weights=use_class_weights,
                                     border_weights_sigma=border_weights_sigma,
                                     shuffle=True)
        
        self.optimizer = self.get_optimizer(self.config.LEARNING_RATE, 
                                            decay=self.config.WEIGHT_DECAY,
                                            momentum=self.config.LEARNING_MOMENTUM,
                                            clipnorm=self.config.GRADIENT_CLIP_NORM)   
        
        # freeze all encoding layers
        for i in range(1,343):
            layer = self.keras_model.layers[i]
            layer.trainable = False
        
        self.loss = self.get_loss()
        self.loss_weights = {'bg_logits': 0.5, 'cl_logits': 0.0}
        self.metrics = self.get_metrics()
        self.keras_model.compile(optimizer=self.optimizer, 
                                 loss=self.loss, 
                                 loss_weights=self.loss_weights,
                                 metrics=self.metrics)
        
        log_dir, checkpoint_dir, initial_epoch = \
            utils_keras.get_log_dir(self.model_name, self.model_dir, self.weights_path)
        callbacks = self.get_callbacks(log_dir, checkpoint_dir)
        
        lr = self.config.LEARNING_RATE
        print("\nStarting at epoch {}. lr={}\n".format(initial_epoch, lr))
        print("Checkpoint Path: {}".format(checkpoint_dir))
        
        self.keras_model.fit_generator(generator=train_generator,
                                       epochs=epochs,
                                       steps_per_epoch=self.config.STEPS_PER_EPOCH,
                                       validation_data=val_generator,
                                       validation_steps=self.config.VALIDATION_STEPS,
                                       initial_epoch=initial_epoch,
                                       callbacks=callbacks,
                                       max_queue_size=100,
                                       workers=multiprocessing.cpu_count(),
                                       use_multiprocessing=False
                                      )
  
    def detect(self, images, preprocessor=None, verbose=0, **kwargs):
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"
        if verbose:
            print("Processing {} images".format(len(images)))
            for image in images:
                print("image: " + image)
        # Mold inputs to format expected by the neural network
        if preprocessor:
            images = [preprocessor(x, kwargs) for x in images]
        batch_images = np.stack(images, axis=0)
        bg_logits, cl_logits = self.keras_model.predict([batch_images], verbose=0)
        
        bg_scores = utils_image.softmax(bg_logits)
        cl_scores = utils_image.softmax(cl_logits)

        labels = np.argmax(bg_scores, axis=-1) * (np.argmax(cl_probs, axis=-1) + 1)
        return [_ for _ in labels], [_ for _ in bg_scores], [_ for _ in cl_scores]

###########################################################################################################################    
    
class DataSequence(Sequence):
    '''Generates data to be used in training model'''
    def __init__(self, dataset, batch_size, config, repeat=1,
                 shuffle=True, **kwargs):
        self.dataset = dataset
        self.image_ids = dataset.image_ids[:]
        self.config = config
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle
        self.on_epoch_end()
        
        self.use_border_weights = kwargs.setdefault("use_border_weights", True)
        self.use_class_weights = kwargs.setdefault("use_class_weights", True)
        self.border_weights_sigma = kwargs.setdefault(
            "border_weights_sigma", max(config.MODEL_INPUT_SIZE)/16)

    def __len__(self):
        ## using np.floor will cause self.on_epoch_end to not be called at the end of
        ## each epoch
        return int(np.ceil(1.0 * len(self.image_ids) * self.repeat / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data. """
        # Generate indexes of the batch
        start = index * self.batch_size % len(self.image_ids)
        end = start + self.batch_size
        image_ids = self.image_ids[start:end]
        ## Re-shuffle the data and fill remaining zero entries
        if end > len(self.image_ids):
            if self.shuffle:
                np.random.shuffle(self.image_ids)
            image_ids = image_ids + self.image_ids[:end-len(self.image_ids)]
        return self.__data_generator(image_ids)
    
    def preprocess(self, *image_inputs):
        """ Modifies input images for training sequence"""
    # only flips vertically for now, more changes to be added later
        for image in image_inputs:
            image = np.flipud(image)
             
    def on_epoch_end(self):
        """ Updates indexes after each epoch. """
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __data_generator(self, image_ids):
        """batch_image: [batch_size, h, w, 3]"""
        # Init the matrix
        # [None, h, w, channel]
        random_val = random.random()
        batch_image, batch_bg_masks, batch_cl_masks = [], [], []
        for image_id in image_ids:
            # access image and mask png files as numpy arrays
            image, (masks, class_ids) = self.dataset.load_image(image_id)
            # randomly select whether batch images are preprocessed
            if random_val < 0.5:
                self.preprocess(image, masks)
            batch_image.append(image)
            
            # check if mask pixel is bg if rgb values are [0,0,0] 
            bg_target = to_categorical(np.any(masks, axis=-1), num_classes=2)
            if self.use_border_weights:
                bg_scores = unet_utils.unet_border_weights(masks, sigma=self.border_weights_sigma)
                bg_target = bg_target * np.expand_dims(bg_scores, axis=-1)
            batch_bg_masks.append(bg_target)  
                       
            # for debugging only
            categories = np.dot(masks,class_ids)
            #print(image_id, str(np.nonzero(categories)[0].size))
            #print(categories[np.nonzero(categories)])
                
            # classify mask pixel by following map: (0, 1, 2, 3, 4) = (bg, stroma, tumor, lymphocyte, blood)
            cl_target = to_categorical(np.dot(masks, class_ids), 
                                      num_classes=self.config.NUM_CLASSES)[:,:,1:] # don't keep bg
            if self.use_class_weights:
                cl_scores = unet_utils.unet_class_weights(masks, class_ids)
                cl_target = cl_target * cl_scores
            batch_cl_masks.append(cl_target)

        return (np.asarray(batch_image), [np.asarray(batch_bg_masks), np.asarray(batch_cl_masks)])
            
        '''
        batch_images = np.zeros((self.batch_size,) + self.config.MODEL_INPUT_SIZE + (3, ),
                                dtype=np.float32)
        # [None, h, w, 2]
        batch_bg_masks = np.zeros((self.batch_size,) + self.config.MODEL_INPUT_SIZE + (2,), 
                                  dtype=np.float32)
        # [None, h, w, nb_classes-1], remove background 0
        batch_cl_masks = np.zeros((self.batch_size,) + self.config.MODEL_INPUT_SIZE + 
                                  (self.config.NUM_CLASSES-1,), dtype=np.float32)
        
        for i in range(len(image_ids)):
            image, (masks, class_ids) = self.dataset.load_data(image_ids[i])

            bg_target = to_categorical(np.any(masks, axis=-1), num_classes=2)
            if self.use_border_weights:
                bg_scores = unet_utils.unet_border_weights(masks, sigma=self.border_weights_sigma)
                bg_target = bg_target * np.expand_dims(bg_scores, axis=-1)

            cl_target = to_categorical(np.dot(masks, class_ids), 
                                       num_classes=self.config.NUM_CLASSES)[:,:,1:]
            if self.use_class_weights:
                cl_scores = unet_utils.unet_class_weights(masks, class_ids)
                cl_target = cl_target * cl_scores

            batch_images[i] = image
            batch_bg_masks[i] = bg_target
            batch_cl_masks[i] = cl_target

        return ([batch_images], [batch_bg_masks, batch_cl_masks])
    '''
        
class dataset():
    # (1, 2, 3, 4) = (stroma, tumor, lymphocyte, blood)
    class_colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [255, 0, 255]}
    
    def __init__(self, config, input_folder):
        self.image_dict = dict() #stores file names of all images
        self.masks_dict = dict() #stores file names all masks
        self.image_ids = os.listdir(input_folder) #stores image IDs
        #access sample folders (image, mask) from input folder specified by ID
        for image_id in self.image_ids:
            image, mask, mask_no_bg = os.listdir(input_folder + image_id)
            #add image and mask to respective lists in dataset
            self.image_dict[image_id] = input_folder + image_id + "/" + image
            self.masks_dict[image_id] = input_folder + image_id + "/" + mask_no_bg

    def load_image(self, image_id):
        '''Render image and mask from folder as numpy arrays'''
        image = io.imread(self.image_dict[image_id])
        masks = io.imread(self.masks_dict[image_id])
        # image, mask: (500,500,3) -> (512,512,3)
        image = np.pad(image,pad_width=((6,6),(6,6),(0,0)),mode='constant')
        masks = np.pad(masks,pad_width=((6,6),(6,6),(0,0)),mode='constant')
        #TO DO: convert mask rgb to binary
        #preprocess_parameters = random_generate_some_pars
        #image = preprocess(image)
        #masks = preprocess(masks)
        class_ids = list(self.class_colors)[:3]
        for i in range(len(class_ids)):
            class_ids[i] /= 255
        return image, (masks, class_ids)       
    
    def remove_background(self, output_dir, image_ids=None):
        '''Output mask png files with background removed for specified images'''
        class_colors = [np.asarray(self.class_colors[cl]) for cl in self.class_colors]
        # set default to all images if none given
        if image_ids is None:
            image_ids = self.image_ids
        # traverse over all given images
        for image_id in image_ids:
            masks = io.imread(self.masks_dict[image_id])
            row, col = masks.shape[:2] # get number of rows and columns
            for r in range(row): # iterate over rows
                for c in range(col): # and columns
                    rgb_val = masks[r,c]
                    color_match = False
                    # check if rgb values match with any class color
                    for color in class_colors:
                        if np.all(rgb_val==color):
                            color_match = True
                    # otherwise change values to all zero
                    if not color_match:
                        masks[r,c] = np.array([0, 0, 0])
            io.imsave(output_dir + ("/" + image_id)*2 + "_mask_no_bg.png", masks)
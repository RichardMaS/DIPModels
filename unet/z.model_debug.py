from keras import backend as K
from keras import callbacks
from keras.applications import imagenet_utils
from keras.engine.topology import get_source_inputs
from keras.layers import Activation
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
from keras.models import Model
from keras.utils import to_categorical
import keras.optimizers
import numpy as np
import tensorflow as tf
import skimage
import time
from scipy import ndimage

from DIPModels.utils_g import image as utils_image
from DIPModels.utils_g import keras as utils_keras
from DIPModels.utils_g import misc as utils_misc


def conv_block(x, filter_size, dropout_rate, use_batch_norm, name):
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    
    block = Conv2D(filter_size, kernel_size=(3, 3), padding='same', name=name + '_1_conv')(x)
    if use_batch_norm:
        block = BatchNormalization(axis=bn_axis, name=name + '_1_bn')(block)
    block = Activation('relu', name=name + '_1_activation')(block)
    block = Conv2D(filter_size, kernel_size=(3, 3), padding='same', name=name + '_2_conv')(block)
    if use_batch_norm:
        block = BatchNormalization(axis=bn_axis, name=name + '_2_bn')(block)
    block = Activation('relu', name=name + '_2_activation')(block)
    if dropout_rate > 0:
        block = Dropout(dropout_rate, name=name + '_1_dp')(block)
    return block

def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layres
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


def unet_2d(x, filter_size, layer, dropout_rate=0.0, use_batch_norm=True):
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    
    conv_layers = []
    for i in range(layer-1):
        name = 'conv_' + str(i+1)
        conv = conv_block(x, (2**i)*filter_size, dropout_rate, use_batch_norm, name=name)
        conv_layers.append(conv)
        x = MaxPooling2D(pool_size=(2, 2), name=name + '_pool')(conv)
    
    x = conv_block(x, 2**(layer-1)*filter_size, dropout_rate, use_batch_norm, name='conv_' + str(layer))
    
    
    _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                             stage5=True, train_bn=config.TRAIN_BN)
    
    
    
    
    
    
    
    
    
    for i in range(layer-2, -1, -1):
        name = 'up_' + str(i+1)
        up = Concatenate(axis=bn_axis, name=name + '_concat')([UpSampling2D(size=(2, 2))(x), conv_layers[i]])
        x = conv_block(up, (2**i)*filter_size, dropout_rate, use_batch_norm, name=name)
    
    return x

def unet_2d_basic(x, filter_size, dropout_rate=0.0, use_batch_norm=True):
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    
    conv_1 = conv_block(x, 1*filter_size, dropout_rate, use_batch_norm, name='conv_1')
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='conv_1_pool')(conv_1)

    conv_2 = conv_block(pool_2, 2*filter_size, dropout_rate, use_batch_norm, name='conv_2')
    pool_3 = MaxPooling2D(pool_size=(2, 2), name='conv_2_pool')(conv_2)

    conv_3 = conv_block(pool_3, 4*filter_size, dropout_rate, use_batch_norm, name='conv_3')
    pool_4 = MaxPooling2D(pool_size=(2, 2), name='conv_3_pool')(conv_3)

    conv_4 = conv_block(pool_4, 8*filter_size, dropout_rate, use_batch_norm, name='conv_4')
    pool_5 = MaxPooling2D(pool_size=(2, 2), name='conv_4_pool')(conv_4)

    conv_5 = conv_block(pool_5, 16*filter_size, dropout_rate, use_batch_norm, name='conv_5')
    pool_6 = MaxPooling2D(pool_size=(2, 2), name='conv_5_pool')(conv_5)

    conv_6 = conv_block(pool_6, 32*filter_size, dropout_rate, use_batch_norm, name='conv_6')

    up_5 = Concatenate(axis=bn_axis, name='up_5_concat')([UpSampling2D(size=(2, 2))(conv_6), conv_5])
    up_conv_5 = conv_block(up_5, 16*filter_size, dropout_rate, use_batch_norm, name='up_5')

    up_4 = Concatenate(axis=bn_axis, name='up_4_concat')([UpSampling2D(size=(2, 2))(up_conv_5), conv_4])
    up_conv_4 = conv_block(up_4, 8*filter_size, dropout_rate, use_batch_norm, name='up_4')

    up_3 = Concatenate(axis=bn_axis, name='up_3_concat')([UpSampling2D(size=(2, 2))(up_conv_4), conv_3])
    up_conv_3 = conv_block(up_3, 4*filter_size, dropout_rate, use_batch_norm, name='up_3')

    up_2 = Concatenate(axis=bn_axis, name='up_2_concat')([UpSampling2D(size=(2, 2))(up_conv_3), conv_2])
    up_conv_2 = conv_block(up_2, 2*filter_size, dropout_rate, use_batch_norm, name='up_2')

    up_1 = Concatenate(axis=bn_axis, name='up_1_concat')([UpSampling2D(size=(2, 2))(up_conv_2), conv_1])
    up_conv_1 = conv_block(up_1, 1*filter_size, dropout_rate, use_batch_norm, name='up_1')
    
    ## return image map without logits and scores
    return up_conv_1


def preprocess_input(x, data_format=None):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format, mode='torch')


class UNet(object):
    """Encapsulates the UNet model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self):
        """
            config: A Sub-class of the Config class
            model_dir: Directory to save training logs and trained weights
        """
        pass

    def __call__(self, config, model_dir, weights_path=None, **kwargs):
        self.build_model(config, model_dir, weights_path, **kwargs)
        return self
    
    def build_model(self, config, model_dir, weights_path, **kwargs):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        self.config = config
        self.weights_path = weights_path
        self.model_dir = model_dir
        self.model_name = config.NAME + '_unet'
        self.nb_classes = config.NUM_CLASSES
        self.nb_layers = config.NUM_LAYERS
        self.filter_size = config.FILTER_SIZE
        
        # Image size must be dividable by 2 multiple times
        h, w = config.MODEL_INPUT_SIZE
        if h % 2**self.nb_layers or w % 2**self.nb_layers:
            raise Exception("Image size must be dividable by " + str(2**self.nb_layers) +
                            "to avoid fractions when downscaling and upscaling.")
        
        img_input = Input(shape=(h, w, 3), name='input')
        conv_final = unet_2d(img_input, 
                             filter_size=config.FILTER_SIZE, 
                             layer=config.NUM_LAYERS,
                             dropout_rate=config.DROP_OUT_RATE, 
                             use_batch_norm=config.USE_BN)
        
        ## Get logits and set background to 0
        bg_logits = Conv2D(2, (1, 1), name='bg_logits')(conv_final)
        #if self.nb_classes > 2:
        cl_logits = Conv2D(self.nb_classes-1, (1, 1), name='cl_logits')(conv_final)
        
        model = Model(inputs=[img_input], outputs=[bg_logits, cl_logits], 
                      name='UNet2D_' + str(config.FILTER_SIZE))
        
        if self.weights_path is not None:
            print("Loading weights from: " + weights_path)
            model.load_weights(weights_path)
            # print("Loading weights from: " + weights_path)
        
        # TODO: Add multi-GPU support.
        #if config.GPU_COUNT > 1:
        #    from parallel_model import ParallelModel
        #    model = ParallelModel(model, config.GPU_COUNT)
        
        self.keras_model = model
    
    def get_optimizer(self, learning_rate, decay=1e-6, momentum=0.9, clipnorm=5.0):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, 
                                         clipnorm=clipnorm, nesterov=True)
        optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                                          epsilon=None, decay=decay, amsgrad=False)
        return optimizer
        
    def get_loss(self):
        def loss_function(target, output):
            
            return K.mean(K.categorical_crossentropy(target, K.softmax(output)))
        return {'bg_logits': loss_function, 'cl_logits': loss_function}
    
    def get_metrics(self):
        def bg_iou_metrics(target, output):
            y_true, y_pred = target, K.softmax(output)
            return K.mean(utils_keras.iou_coef(y_true, y_pred))
        
        def cl_iou_metrics(target, output):
            y_true, y_pred = target, K.softmax(output)
            # Set bg region to 0 for output
            y_pred = tf.where(tf.not_equal(y_true, tf.constant(0., dtype=tf.float32)), 
                              y_pred, y_true)
            return K.mean(utils_keras.iou_coef(y_true, y_pred))
        
        return {'bg_logits': bg_iou_metrics, 'cl_logits': cl_iou_metrics}
    
    def get_callbacks(self, log_dir, checkpoint_dir):
        callbacks_list = [callbacks.ModelCheckpoint(checkpoint_dir, 
                                                    verbose=0, save_weights_only=True),
                          callbacks.TensorBoard(log_dir, histogram_freq=0,
                                                write_graph=False, write_images=False)]
        return callbacks_list
        
    def train(self, train_dataset, val_dataset, epochs,
              use_border_weights=False, use_class_weights=False):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.

        """
        # Data generators
        train_generator = data_generator(train_dataset, self.config.BATCH_SIZE,
                                         model_input_shape=self.config.MODEL_INPUT_SIZE, 
                                         nb_classes=self.config.NUM_CLASSES,
                                         use_border_weights=use_border_weights,
                                         use_class_weights=use_class_weights,
                                         shuffle=True)
        val_generator = data_generator(val_dataset, self.config.BATCH_SIZE,
                                       model_input_shape=self.config.MODEL_INPUT_SIZE, 
                                       nb_classes=self.config.NUM_CLASSES,
                                       use_border_weights=use_border_weights,
                                       use_class_weights=use_class_weights,
                                       shuffle=True)
        
        self.optimizer = self.get_optimizer(self.config.LEARNING_RATE, 
                                            decay=self.config.WEIGHT_DECAY,
                                            momentum=self.config.LEARNING_MOMENTUM,
                                            clipnorm=self.config.GRADIENT_CLIP_NORM)
        
        self.loss = self.get_loss()
        self.loss_weights = {'bg_logits': 0.5, 'cl_logits': 0.0}
        self.metrics = self.get_metrics()
        self.keras_model.compile(optimizer=self.optimizer, 
                                 loss=self.loss, 
                                 loss_weights=self.loss_weights,
                                 metrics=self.metrics)
        
        log_dir, checkpoint_dir, initial_epoch = utils_keras.get_log_dir(self.model_name, self.model_dir, self.weights_path)
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
                                       workers=max(self.config.BATCH_SIZE // 2, 2),
                                       use_multiprocessing=True
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


## TODO: modify data_generator to support multiple labels, parse NB_CLASSES here
def data_generator(dataset, batch_size, model_input_shape, nb_classes, 
                   shuffle=True, use_border_weights=True, use_class_weights=True,
                   preprocessor=None, **kwargs):
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    
    b = 0
    while True:
        image_index = (image_index + 1) % len(image_ids)
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)
        image_id = image_ids[image_index]
        #print("For %s: " % image_id)
        #start_time = time.time()
        image, (masks, class_ids) = dataset.load_data(image_id)
        #print("Time for dataset.load_data: --- % s seconds ---" % (time.time() - start_time))
        if preprocessor is not None:
            #start_time = time.time()
            image = preprocessor(image, **kwargs)
            #print("Time for preprocess image: --- % s seconds ---" % (time.time() - start_time))
            #start_time = time.time()
            masks = preprocessor(masks, **kwargs)
            #print("Time for preprocess masks: --- % s seconds ---" % (time.time() - start_time))
        
        bg_target = to_categorical(np.any(masks, axis=-1), num_classes=2)
        if use_border_weights:
            bg_scores = unet_border_weights(masks, sigma=16)
            bg_target = bg_target * np.expand_dims(bg_scores, axis=-1)
        
        cl_target = to_categorical(np.dot(masks, class_ids), num_classes=nb_classes)[:,:,1:]
        if use_class_weights:
            cl_scores = unet_class_weights(masks, class_ids)
            cl_target = cl_target * cl_scores
        
        if b == 0:
            # [None, h, w, channel]
            batch_images = np.zeros((batch_size,) + (model_input_shape[0], model_input_shape[1], 3),
                                    dtype=np.float32)
            # [None, h, w, 2]
            batch_bg_masks = np.zeros((batch_size, ) + model_input_shape + (2,), dtype=np.float32)
            
            # [None, h, w, nb_classes-1], remove background 0
            batch_cl_masks = np.zeros((batch_size, ) + model_input_shape + (nb_classes-1,),
                                      dtype=np.float32)
            
        batch_images[b] = image
        batch_bg_masks[b] = bg_target
        batch_cl_masks[b] = cl_target
        b += 1
        
        if b >= batch_size:
            b = 0
            # print("yield batch ---")
            yield ([batch_images], [batch_bg_masks, batch_cl_masks])

            
def unet_border_weights(masks, w0=1, sigma=4, r=3):
    ## ref : https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34/notebook
    n_masks = masks.shape[-1]
    inner_masks = np.stack([skimage.morphology.binary_erosion(masks[:,:,x], skimage.morphology.disk(r)) 
                            for x in range(n_masks)], axis=-1)
    outer_masks = np.stack([skimage.morphology.binary_dilation(masks[:,:,x], skimage.morphology.disk(r)) 
                            for x in range(n_masks)], axis=-1)
    border = np.logical_xor(outer_masks, inner_masks)

    # calculate weight for important pixels
    distances = np.array([ndimage.distance_transform_edt(border[:,:,x] == 0) 
                          for x in range(n_masks)])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)
    d2 = np.zeros(d1.shape)
    weights = w0 * np.exp(-(d1+d2)**2/(2*sigma**2)).astype(np.float32)
    #weights[weights < 1e-4] = 0.
    
    # Set all positive to 1
    #labels = utils_misc.masks_to_labels(masks, binary=True)
    #weights[labels > 0] = 1
    return weights

def unet_class_weights(masks, class_ids):
    n_masks = masks.shape[-1]
    radius = [np.sqrt(np.sum(masks[:,:,x])) for x in range(n_masks)]
    weights = np.zeros((max(class_ids),))
    for r, class_id in zip(radius, class_ids):
        weights[class_id-1] += r
    return weights/np.sum(weights)

def crossentropy_with_zero_padding(y_true, y_pred):
    ## logits = [bg_logits, cl_1_logits, cl_2_logits, ... cl_n-1_logits]
    logits = K.concatenate([y_pred, K.zeros_like(y_pred[..., :1])], axis=-1)
    return K.categorical_crossentropy(y_true, logits, from_logits=True)


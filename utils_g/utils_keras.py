import datetime
import os
import re
import tensorflow as tf
import numpy as np

from keras import backend as K
from keras import callbacks
from keras import losses
from sklearn.metrics import roc_auc_score
import keras.preprocessing.image as KP_image

IMAGE_NET_MEAN = np.array([123.68, 116.779, 103.939])

class roc_callback(callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
    
    def on_train_begin(self, logs={}):
        return
    
    def on_train_end(self, logs={}):
        return
    
    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' %
              (str(round(roc, 4)), str(round(roc_val, 4))))
        return

    def on_batch_begin(self, batch, logs={}):
        return
    
    def on_batch_end(self, batch, logs={}):
        return

def dice_coef(y_true, y_pred):
    """ Calculate dice coefficient for y_true ad y_pred
        y_true: [batch_size, h, w, nb_classes]
        y_pred: [batch_size, h, w, nb_classes]
        
        Return: dice_coef for each classes. [batch_size, nb_classes]
                Apply weight to each classes, use 0 to ignor background
                weights = tf.convert_to_tensor([0, ...], dice_coef.dtype.base_dtype)
                dice_coef *= weights / tf.reduce_sum(weights)
    """
    nb_classes = y_pred.get_shape()[-1]
    y_true = K.one_hot(tf.argmax(y_true, axis=-1), num_classes=nb_classes)
    y_pred = K.one_hot(tf.argmax(y_pred, axis=-1), num_classes=nb_classes)

    axis = np.arange(1, K.ndim(y_pred)-1)
    intersect = K.sum(y_true * y_pred, axis=axis)
    union = K.sum(y_true + y_pred, axis=axis) - intersect
    dice_coef =  2.0 * intersect/(union + intersect + K.epsilon())
    
    return dice_coef

def iou_coef(y_true, y_pred, exclude_bg=True):
    """ Calculate IoU coefficient for y_true ad y_pred
        y_true: [batch_size, h, w, nb_classes]
        y_pred: [batch_size, h, w, nb_classes]
        
        Return: iou_coef for each classes. [batch_size, nb_classes (-1)]
                Apply weight to each classes, use 0 to ignor background
                weights = tf.convert_to_tensor([0, ...], dice_coef.dtype.base_dtype)
                dice_coef *= weights / tf.reduce_sum(weights)
    """
    nb_classes = y_pred.get_shape()[-1]
    y_true = K.one_hot(tf.argmax(y_true, axis=-1), num_classes=nb_classes)
    y_pred = K.one_hot(tf.argmax(y_pred, axis=-1), num_classes=nb_classes)
    if exclude_bg:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]

    axis = np.arange(1, K.ndim(y_pred)-1)
    intersect = K.sum(y_true * y_pred, axis=axis)
    union = K.sum(y_true + y_pred, axis=axis) - intersect
    iou_coef = 1.0 * intersect/(union + K.epsilon())
    
    return iou_coef

def weighted_categorical_crossentropy(target, output, weights=None):
    # Note: Keras expects probabilities.
    if weights is not None:
        weights = tf.convert_to_tensor(weights, output.dtype.base_dtype)
        # assert weights.get_shape()[0] == target.get_shape()[-1], "dimension do not match"
        target = target * (weights / tf.reduce_sum(weights))
    return losses.categorical_crossentropy(target, output)

def weighted_pixelwise_crossentropy(target, output, weights=None):
    if weights is not None:
        weights = np.expand_dims(weights, axis=-1)
        weights = tf.convert_to_tensor(weights, output.dtype.base_dtype)
        target = target * weights
    return losses.categorical_crossentropy(target, output)

def detect_single_image_with_flipping(predict_func, x, types,
                                      concat_d, **kwargs):
    """ Predict result by combining results from horizontal/vertical flipping.
        
        For each image A, consider A, A[:,::-1], A[::-1,:], A[::-1, ::-1]
        and their transpose. The function combine the prediction result of
        the above 8 images. Currently requires predict_func to organize results
        into numpy array, or numpy array nested in tuple and dictionary.
        
        Arguments:
            predict_func: predict function used to predict a single image/batch.
            x: image
            types: one of ('scalar', 'image', 'coord', 'boxes'), output will
                   be concat along batch dimension (axis=0) by default.
            concat_d: User specified concat dimension.
            kwargs: other parameters passed to predict_func
        
        Return:
            An object which has same structure as predict_func by combining all
            results from flipping.
        
        Examples:
            For Mask_RCNN: totally N masks
                rois: bounding box [x1, y1, x2, y2], shape=(N, 4)
                class_ids: class_ids(all 1 in our case), shape=(N,)
                scores: scores for mask, shape=(N,)
                masks: single mask image, shape=(x.height, x.width, N)
            Call:
                detect_single_image_with_flipping(model.detect, image,
                    types=dict(class_ids='scalar', scores='scalar',
                               masks='image', rois='boxes'),
                    concat_d=dict(class_ids=0, scores=0, masks=-1, rois=0))
    """
    h, w = x.shape[0], x.shape[1]
    batch_list = [x, np.transpose(x, (1, 0, 2)),
                  x[:, ::-1, :], np.transpose(x[:, ::-1, :], (1, 0, 2)),
                  x[::-1, :, :], np.transpose(x[::-1, :, :], (1, 0, 2)),
                  x[::-1, ::-1, :], np.transpose(x[::-1, ::-1, :], (1, 0, 2))]
    result_list = [predict_func([_], **kwargs)[0] for _ in batch_list]
    
    if not result_list:
        return None

    def _invert(x, flag):
        if flag == 'image':
            return [x[0], np.transpose(x[1], (1, 0, 2)),
                    x[2][:, ::-1, :], np.transpose(x[3], (1, 0, 2))[:, ::-1, :],
                    x[4][::-1, :, :], np.transpose(x[5], (1, 0, 2))[::-1, :, :],
                    x[6][::-1, ::-1, :],
                    np.transpose(x[7], (1, 0, 2))[::-1, ::-1, :]]
        elif flag == 'boxes':
            return [x[0], x[1][:,[1,0,3,2]],
                    x[2][:,[0,3,2,1]] * np.array([1, -1, 1, -1]) + np.array([0, w-1, 0, w-1]),
                    x[3][:,[1,2,3,0]] * np.array([1, -1, 1, -1]) + np.array([0, w-1, 0, w-1]),
                    x[4][:,[2,1,0,3]] * np.array([-1, 1, -1, 1]) + np.array([h-1, 0, h-1, 0]),
                    x[5][:,[3,0,1,2]] * np.array([-1, 1, -1, 1]) + np.array([h-1, 0, h-1, 0]),
                    -x[6][:,[2,3,0,1]] + np.array([h-1, w-1, h-1, w-1]),
                    -x[7][:,[3,2,1,0]] + np.array([h-1, w-1, h-1, w-1])]
        elif flag == 'coord':
            pass
        else:
            return x

    ## Revert image and coordinate back to origin
    if isinstance(result_list[0], (dict)):
        keys = result_list[0].keys()
        res = dict()
        for k in keys:
            tmp = _invert([_[k] for _ in result_list], flag=types[k])
            res[k] = np.concatenate(tmp, axis=concat_d[k])
        return res
    elif isinstance(result_list[0], (list, tuple)):
        N = len(result_list[0])
        res = [None] * N
        for k in range(N):
            tmp = _invert([_[k] for _ in result_list], flag=types[k])
            res[k] = np.concatenate(tmp, axis=concat_d[k])
        return res
    else:
        return np.concatenate(result_list, axis=concat_d)

def detect_image_with_assemble(predict_func, x, target_size, **kwargs):
    
    target_size
    
    
    
    
def get_log_dir(model_name=None, model_dir=None, weights_path=None):
    """ Sets the model log directory and epoch counter.

        model_dir: specify where to write the model, the folder will be 
                   extended to model_dir/yyyymmddThhmm
        prev_weights_path: If specified, model will write new model into its 
                           folder and use the epoch of this file. 
    """
    # Set date and epoch counter as if starting a new model
    if model_dir:
        epoch = 0
        now = datetime.datetime.now()
        log_dir = os.path.join(model_dir, "{}{:%Y%m%dT%H%M}".format(model_name, now))
    else:
        if weights_path:
            # weights_path = '/path/to/logs/yyyymmddThhmm/model_name_0020.h5
            # regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/(\w+)\_(\d{4})\.h5"
            regex = r"(.*)/(\w+)\_(\d{4})\.h5"
            m = re.match(regex, weights_path)
            if not m:
                raise ValueError("weights_path need to be like model_name_0020.h5")
            log_dir, model_name, epoch = m.group(1), m.group(2), int(m.group(3))
        else:
            raise ValueError("model_dir and weights_path cannot both be None")
    
    checkpoint_dir = os.path.join(log_dir, "{}_*epoch*.h5".format(model_name))
    checkpoint_dir = checkpoint_dir.replace("*epoch*", "{epoch:04d}")
    
    return log_dir, checkpoint_dir, epoch

def load_pretrained_weights(model, weights_path, layer_indices):
    f = h5py.File(weights_path, mode='r')
    weights = [f['graph']['param_{}'.format(p)] for p in layer_indices]
    model.set_weights(weights)
    f.close()


def data_generator(train_processor, valid_processor, train_dir, valid_dir,
                   batch_size, class_mode,
                   target_size=None, save_to_dir=None, **kwargs):
    # Training generator with augmentation
    train_datagen = KP_image.ImageDataGenerator(preprocessing_function=lambda x: train_processor(x, kwargs))
    train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                        target_size=target_size,
                                                        save_to_dir=save_to_dir,
                                                        batch_size=batch_size,
                                                        class_mode=class_mode)
    
    # Validation generator without augmentation
    valid_datagen = image.ImageDataGenerator(preprocessing_function=lambda x: valid_processor(x, kwargs))
    valid_generator = valid_datagen.flow_from_directory(directory=valid_dir,
                                                        target_size=target_size,
                                                        save_to_dir=save_to_dir,
                                                        batch_size=batch_size,
                                                        class_mode=class_mode)
    return train_generator, valid_generator


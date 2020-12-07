""" Add validation metrics support to MaskRCNN class (mrcnn/model.py). """
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.utils as KU

import tensorflow as tf
import numpy as np
import os
import multiprocessing

from .mrcnn import model as model_o
from .mrcnn import utils as utils_o

def get_masks_true_and_pred(target_masks, target_class_ids, pred_masks):
    """ Extract ROIs and Masks for mask loss and evaluation index.
        
        target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32
        tensor with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])
    
    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)
    
    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)
                           
    return y_true, y_pred

def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """ Mask binary cross-entropy loss for the masks head.
        
        target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32
        tensor with values from 0 to 1.
    """
    y_true, y_pred = get_masks_true_and_pred(
        target_masks, target_class_ids, pred_masks)
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss

def dice_coef_graph(target_masks, target_class_ids, pred_masks):
    """ Dice coefficient for masks head.
        
        target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32
        tensor with values from 0 to 1.
    """
    y_true, y_pred = get_masks_true_and_pred(
        target_masks, target_class_ids, pred_masks)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(tf.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    
    return (2.0 * intersection) / (union + intersection + 1e-8)

def iou_coef_graph(target_masks, target_class_ids, pred_masks):
    """ IOU coefficient/Jaccard for masks head.
        
        target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32
        tensor with values from 0 to 1.
        """
    y_true, y_pred = get_masks_true_and_pred(
        target_masks, target_class_ids, pred_masks)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(tf.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return intersection / (union + 1e-8)


############################################################
#  MaskRCNN Class
############################################################
## TODO: debug multi_gpu support. Both keras.utils.multi_gpu_model 
## and mrcnn/ParallelModel don't work. 
class MaskRCNN(model_o.MaskRCNN):
    """ Inherit from mrcnn/model.py MaskRCNN class.
        Add extra evaluation metrics and multi_gpu support.
        The actual Keras model is in the keras_model property.
    """
    def __init__(self, mode, config, model_dir):
        """
            mode: Either "training" or "inference"
            config: A Sub-class of the Config class
            model_dir: Directory to save training logs and trained weights
            """
        super(self.__class__, self).__init__(mode, config, model_dir)
        self.add_metrics()

    def add_metrics(self):
        """ Add dice coef and iou to output layers. """
        if self.mode != 'training':
            return
        
        # get the keras_model from MaskRCNN class
        use_multiGPU = hasattr(self.keras_model, "inner_model")
        model = (self.keras_model.inner_model if use_multiGPU 
                 else self.keras_model)
        
        # calculate dice_coef and iou_coef
        target_mask, target_class_ids, mrcnn_mask = \
            model.get_layer('mrcnn_mask_loss').input
        dice_coef = KL.Lambda(lambda x: dice_coef_graph(*x), name="dice_coef")(
            [target_mask, target_class_ids, mrcnn_mask])
        iou_coef = KL.Lambda(lambda x: iou_coef_graph(*x), name="iou_coef")(
            [target_mask, target_class_ids, mrcnn_mask])
        
        # Rebuild keras_model with original inputs and outputs + iou/dice
        model = KM.Model(model.inputs, model.outputs + [dice_coef, iou_coef],
                         name='mask_rcnn')
        
        # use the mrcnn.parallel_model/ParallelModel for multi_gpu
        if use_multiGPU:
            from .mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, self.keras_model.gpu_count)
        self.keras_model = model
        return

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None):
        """ Use keras Sequence as data generator. """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = DataSequence(train_dataset, self.config, shuffle=True,
                                       augmentation=augmentation,
                                       batch_size=self.config.BATCH_SIZE)
        val_generator = DataSequence(val_dataset, self.config, shuffle=True,
                                     batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Train
        model_o.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        model_o.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)
    
    
    def compile(self, learning_rate, momentum):
        """ Add dice-coef and iou into metrics. """
        super(self.__class__, self).compile(learning_rate, momentum)
        
        # Add dice-coef and iou metrics
        metrics_names = ["dice_coef", "iou_coef"]
        for name in metrics_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(
                tf.reduce_mean(layer.output, keepdims=True))


class DataSequence(KU.Sequence):
    def __init__(self, dataset, config, shuffle=True, augmentation=None,
                 random_rois=0, batch_size=1, repeat=1, detection_targets=False):
        self.dataset = dataset
        self.config = config
        self.image_ids = np.copy(dataset.image_ids)
        self.random_rois = random_rois
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle
        self.augmentation = augmentation
        # detection_targets is only used for debug, so deprecated here.
        self.detection_targets = detection_targets
        
        if self.config.USE_MINI_MASK:
            self.gt_masks_shape = self.config.MINI_MASK_SHAPE
        else:
            self.gt_masks_shape = (self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1])
        

    def __len__(self):
        ## use np.floor will cause self.on_epoch_end not being called at the end of
        ## each epoch
        return int(np.ceil(1.0 * len(self.image_ids) * self.repeat / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data. """
        # Generate indexes of the batch
        s = index * self.batch_size % len(self.image_ids)
        e = s + self.batch_size
        image_ids = np.copy(self.image_ids[s:e])
        ## Re-shuffle the data and fill 0 entries
        if e > len(self.image_ids):
            if self.shuffle:
                np.random.shuffle(self.image_ids)
            image_ids = np.append(image_ids, self.image_ids[:e-len(self.image_ids)])
    
        return self.__data_generator(image_ids)

    def on_epoch_end(self):
        """ Updates indexes after each epoch. """
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __data_generator(self, image_ids):
        # Anchors: [anchor_count, (y1, x1, y2, x2)]
        backbone_shapes = model_o.compute_backbone_shapes(self.config, self.config.IMAGE_SHAPE)
        anchors = utils_o.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                   self.config.RPN_ANCHOR_RATIOS,
                                                   backbone_shapes,
                                                   self.config.BACKBONE_STRIDES,
                                                   self.config.RPN_ANCHOR_STRIDE)
        
        # Init batch arrays
        batch_image_meta = np.zeros(
            (self.batch_size,) + (self.config.IMAGE_META_SIZE,), dtype=np.float32)
        batch_rpn_match = np.zeros(
            [self.batch_size, anchors.shape[0], 1], dtype=np.int32)
        batch_rpn_bbox = np.zeros(
            [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=np.float32)
        batch_images = np.zeros(
            (self.batch_size,) + tuple(self.config.IMAGE_SHAPE), dtype=np.float32)
        batch_gt_class_ids = np.zeros(
            (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
        batch_gt_boxes = np.zeros(
            (self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
        batch_gt_masks = np.zeros(
            (self.batch_size, self.gt_masks_shape[0], self.gt_masks_shape[1],
             self.config.MAX_GT_INSTANCES), dtype=np.bool)
        
        if self.random_rois:
            batch_rpn_rois = np.zeros(
                (batch_size, self.random_rois, 4), dtype=np.int32)
            if self.detection_targets:
                batch_rois = np.zeros(
                    (batch_size,) + (self.config.TRAIN_ROIS_PER_IMAGE, 4), dtype=np.int32)
                batch_mrcnn_class_ids = np.zeros(
                    (batch_size,) + (self.config.TRAIN_ROIS_PER_IMAGE,), dtype=np.int32)
                batch_mrcnn_bbox = np.zeros(
                    (batch_size,) + (self.config.TRAIN_ROIS_PER_IMAGE, self.config.NUM_CLASSES, 4), dtype=np.float32)
                batch_mrcnn_mask = np.zeros(
                    (batch_size,) + (config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], 
                                     config.MASK_SHAPE[1], config.NUM_CLASSES), dtype=np.float32)
        
        for i in range(len(image_ids)):
            # Get GT bounding boxes and masks for image.
            image_id = image_ids[i]
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                model_o.load_image_gt(self.dataset, self.config, image_id, 
                                      augmentation=self.augmentation,
                                      use_mini_mask=self.config.USE_MINI_MASK)
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue
            
            # RPN Targets
            rpn_match, rpn_bbox = model_o.build_rpn_targets(image.shape, anchors, 
                                                            gt_class_ids, gt_boxes, self.config)
            
            # Mask R-CNN Targets
            if self.random_rois:
                rpn_rois = model_o.generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if self.detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        model_o.build_detection_targets(
                        rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)
            
            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[i] = image_meta
            batch_rpn_match[i] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[i] = rpn_bbox
            batch_images[i] = model_o.mold_image(image.astype(np.float32), self.config)
            batch_gt_class_ids[i, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[i, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[i, :, :, :gt_masks.shape[-1]] = gt_masks
            if self.random_rois:
                batch_rpn_rois[i] = rpn_rois
                if detection_targets:
                    batch_rois[i] = rois
                    batch_mrcnn_class_ids[i] = mrcnn_class_ids
                    batch_mrcnn_bbox[i] = mrcnn_bbox
                    batch_mrcnn_mask[i] = mrcnn_mask

        inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
        outputs = []

        if self.random_rois:
            inputs.extend([batch_rpn_rois])
            if self.detection_targets:
                inputs.extend([batch_rois])
                # Keras requires that output and targets have the same number of dimensions
                batch_mrcnn_class_ids = np.expand_dims(
                    batch_mrcnn_class_ids, -1)
                outputs.extend(
                    [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

        return inputs, outputs

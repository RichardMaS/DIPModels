import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color
import skimage.io
import urllib.request
import shutil
import warnings


class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.image_map = dict()

    def __call__(self, image_dir, processor=None, tf_args=None, img_args=None):
        raise NotImplementedError("Subclass must implement this function")
    
    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
        self.image_map[image_id] = len(self.image_info) - 1

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]
    
    def find_image_id(self, image_id):
        return self.image_map[image_id]
    
    
def crossentropy_with_zero_padding(y_true, y_pred):
    ## logits = [bg_logits, cl_1_logits, cl_2_logits, ... cl_n-1_logits]
    logits = K.concatenate([y_pred, K.zeros_like(y_pred[..., :1])], axis=-1)
    return K.categorical_crossentropy(y_true, logits, from_logits=True)

## depreacted, use utils.Sequence, this generator may not support multiprocessing
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
        image, (masks, class_ids) = dataset.load_data(image_id)
        if preprocessor is not None:
            image = preprocessor(image, **kwargs)
            masks = preprocessor(masks, **kwargs)
        
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
            yield ([batch_images], [batch_bg_masks, batch_cl_masks])

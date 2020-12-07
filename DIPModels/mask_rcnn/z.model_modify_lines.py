"""
Modified lines in model.py
"""
from . import utils

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    # image = dataset.load_image(image_id)
    # mask, class_ids = dataset.load_mask(image_id)
    image, (mask, class_ids) = dataset.load_data(image_id)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    # active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    # source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    # active_class_ids[source_class_ids] = 1
    rss = dataset.images[image_id]["source"]
    active_class_ids = np.array(dataset.get_source_class_ids(rss), dtype=np.int32)
    
    # Image meta data
    idx = dataset.image_id_mapping[image_id]
    image_meta = compose_image_meta(idx, shape, window, active_class_ids)

def build(self, mode, config):
    line_2061: from .mrcnn.parallel_model import ParallelModel

import os
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage
import time

## Default mean/std
IMAGE_NET_MEAN_TF = np.array([123.68, 116.779, 103.939])
IMAGE_NET_STD_TF = 1.0
IMAGE_NET_MEAN_TORCH = np.array([0.485, 0.456, 0.406])
IMAGE_NET_STD_TORCH = np.array([0.225, 0.224, 0.229])


def softmax(logits):
    probs = np.exp(logits - np.amax(logits, axis=-1, keepdims=True))
    return probs/np.sum(probs, axis=-1, keepdims=True)

def pad_image(x, size, is_target=False, fill_mode='reflect', **kwargs):
    if is_target:
        h, w = x.shape[:2]
        est_h, est_w = max(h, size[0]), max(w, size[1])
        pad_width = [(int((est_h-h)/2), int((est_h-h+1)/2)),
                     (int((est_w-w)/2), int((est_w-w+1)/2))]
    else:
        pad_width = [x for x in size]
    if x.ndim == 3:
        pad_width.append((0, 0))
    return skimage.util.pad(x, pad_width, fill_mode, **kwargs)


def crop_image(x, size, is_target=False, random=False, **kwargs):
    """ x in channel last format
        size: the crop size or the target size
        is_target: let the function know whether size is crop_size or keep_size
        """
    if is_target:
        h, w = x.shape[:2]
        est_h, est_w = min(h, size[0]), min(w, size[1])
        if random:
            dh, dw = np.random.randint(h-est_h+1), np.random.randint(w-est_w+1)
        else:
            dh, dw = int((h-est_h)/2), int((w-est_w)/2)
        crop_width = [(dh, h-est_h-dh), (dw, w-est_w-dw)]
    else:
        crop_width = [x for x in size]
    if x.ndim == 3:
        crop_width.append((0, 0))
    return skimage.util.crop(x, crop_width)


def apply_to_channel(x, func, channel_axis=None, args=(), kwargs=dict()):
    """ Apply function to each channel. 
        Use channel_axis=None for data without channel layer,
        like 2D gray image.
    """
    if channel_axis is None:
        return func(x, *args, **kwargs)
    else:
        channel_axis = channel_axis % x.ndim
        x = np.rollaxis(x, channel_axis, 0)
        channels = [func(_, *args, **kwargs) for _ in x]
        x = np.stack(channels, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x


def rescale_channel(x, rescale_method=None, out_range=None, **kwargs):
    """ Rescale one channel with stretch, hist or adaptive. 
        rescale function will put in_range[0] < x < in_range[1] to (0, 1)
        so rescale back to (x.min(), x.max()) to keep rescaled image
        in same scale.
    """
    x = x * 1.0
    if out_range is None:
        out_range = (x.min(), x.max())
    if rescale_method == 'stretch':
        # Contrast stretching: in_range -> (0, 1)
        stretch_range = (2, 98) if 'stretch_range' not in kwargs else kwargs['stretch_range']
        stretch_range = tuple(np.percentile(x, stretch_range))
        x = skimage.exposure.rescale_intensity(x, in_range=stretch_range)
    elif rescale_method == 'hist':
        # Equalization
        x = skimage.exposure.equalize_hist(x)
    elif rescale_method == 'adaptive':
        # Adaptive Equalization
        clip_limit = 0.03 if 'clip_limit' not in kwargs else kwargs['clip_limit']
        x = skimage.exposure.equalize_adapthist(x, clip_limit=clip_limit)
    return skimage.exposure.rescale_intensity(x, out_range=out_range)


def rescale_intensity_with_deconv(x, out_range=(0, 1), **kwargs):
    if x.ndim < 3:
        layer = x
    else:
        if np.all(x[:,:,0] == x[:,:,1]):
            ## rgb2gray transfer x to 0~1 range gray image
            x = skimage.color.rgb2gray(x)
        else:
            ## Deconvolution HED color image
            x = skimage.color.rgb2hed(x)[:, :, 0]
    ## Maybe denoise after deconvolution
    return skimage.exposure.rescale_intensity(x, out_range=out_range)


def random_transform_pars(N, args):
    """ Generate args for random image augmentation.
        If a scalar value is provided, the function will
        randomly generate N parameters inside the range
        If a list/array is provided, the function will use
        all combination of these values.
        
        # Arguments
        seed: random seed.
        
        # Returns
        A dictionary contains args for random transformation.
        Use function random_transform with this result.
        """
    if 'seed' in args and args['seed'] is not None:
        np.random.seed(args['seed'])
    
    rotation = args['rotation']
    width_shift = args['width_shift']
    height_shift = args['height_shift']
    shear = args['shear']
    width_zoom = args['width_zoom']
    height_zoom = args['height_zoom']
    horizontal_flip = args['horizontal_flip']
    vertical_flip = args['vertical_flip']
    
    pars = dict()
    
    if np.isscalar(rotation) and rotation:
        pars['theta'] = np.deg2rad(np.random.uniform(-rotation, rotation, N))
    else:
        pars['theta'] = np.zeros(N)
    
    if np.isscalar(height_shift) and height_shift:
        pars['tx'] = np.random.uniform(-height_shift, height_shift, N)
        if height_shift < 1 and height_shift > 0:
            pars['tx'] *= args['image_size'][0]
    else:
        pars['tx'] = np.zeros(N)
    
    if np.isscalar(width_shift) and width_shift:
        pars['ty'] = np.random.uniform(-width_shift, width_shift, N)
        if width_shift < 1 and width_shift > 0:
            pars['ty'] *= args['image_size'][1]
    else:
        pars['ty'] = np.zeros(N)
    
    if np.isscalar(shear) and shear:
        pars['shear'] = np.deg2rad(np.random.uniform(-shear, shear, N))
    else:
        pars['shear'] = np.zeros(N)
    
    if np.isscalar(height_zoom) and height_zoom:
        pars['zx'] = np.random.uniform(1 - height_zoom, 1 + height_zoom, N)
    else:
        pars['zx'] = np.ones(N)
    
    if np.isscalar(width_zoom) and width_zoom:
        pars['zy'] = np.random.uniform(1 - width_zoom, 1 + width_zoom, N)
    else:
        pars['zy'] = np.ones(N)
    
    if np.isscalar(horizontal_flip) and horizontal_flip:
        pars['horizontal_flip'] = np.random.random(N) < 0.5
    else:
        pars['horizontal_flip'] = np.zeros(N)
    
    if np.isscalar(vertical_flip) and vertical_flip:
        pars['vertical_flip'] = np.random.random(N) < 0.5
    else:
        pars['vertical_flip'] = np.zeros(N)
    
    return [dict(zip(pars.keys(), _)) for _ in zip(*[v for v in pars.values()])]


def product_transform_pars(args):
    key_map = dict(rotation='theta', height_shift='tx', width_shift='ty',
                   shear='shear', width_zoom='zx', height_zoom='zy',
                   horizontal_flip='horizontal_flip',
                   vertical_flip='vertical_flip')
    par_list = [args[k] for k in key_map]
    pars = [dict(zip([key_map[k] for k in key_map], x))
            for x in itertools.product(*par_list)]
        
    for x in pars:
        x['theta'] = np.deg2rad(x['theta'])
        if abs(x['tx']) < 1 and abs(x['tx']) > 0:
            x['tx'] *= args['image_size'][0]
        if abs(x['ty']) < 1 and abs(x['ty']) > 0:
            x['ty'] *= args['image_size'][1]
        x['shear'] = np.deg2rad(x['shear'])
        x['zx'] = 1 - x['zx']
        x['zy'] = 1 - x['zy']

    return pars


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)

    # o_x = float(args['height']) / 2 + 0.5
    # o_y = float(args['width']) / 2 + 0.5
    # matrix = transform_matrix[:2, :2]
    # offset = np.dot(np.identity(2) - matrix, np.array([[o_x], [o_y]])) + transform_matrix[:2, [2]]
    # transform_matrix_self_try = np.vstack((np.hstack((matrix, offset)), np.array([0,0,1])))
    return transform_matrix


def generate_transform_matrix(args):
    # tform = AffineTransform(scale=(zx, zy), rotation=theta, shear=shear, translation=(tx, ty))
    # return tform.params
    ## Calculate transform_matrix
    transform_matrix = None
    if 'theta' in args and args['theta'] != 0:
        rotation_matrix = np.array([[np.cos(args['theta']), -np.sin(args['theta']), 0],
                                    [np.sin(args['theta']), np.cos(args['theta']), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix
    
    if ('tx' in args and args['tx'] != 0) or ('ty' in args and args['ty'] != 0):
        shift_matrix = np.array([[1, 0, args['tx']],
                                 [0, 1, args['ty']],
                                 [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)
    
    if 'shear' in args and args['shear'] != 0:
        shear_matrix = np.array([[1, -np.sin(args['shear']), 0],
                                 [0, np.cos(args['shear']), 0],
                                 [0, 0, 1]])
        transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
    
    if ('zx' in args and args['zx'] != 1) or ('zy' in args and args['zy'] != 1):
        zoom_matrix = np.array([[args['zx'], 0, 0],
                                [0, args['zy'], 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)
    
    return transform_matrix


def transform_image(x, transform_matrix, inv=False, output_shape=None, 
                    horizontal_flip=False, vertical_flip=False, 
                    fill_mode='reflect', cval=0., **kwargs):
    """Transform image with transform matrix
        
        # Arguments
        x: the input 2D image
        args: random transform args generated by generate_random_transform_pars.
        
        # Returns
        A transformed version of the input (same shape) based on args.
    """
    if output_shape is None:
        output_shape = x.shape
        
    if transform_matrix is not None:
        # pad image first in order to reserve all pixels when revert back.
        side_len = int(np.ceil((2*(output_shape[0]**2 + output_shape[1]**2))**0.5/10)*10)
        x = pad_image(x, size=(side_len, side_len), is_target=True, fill_mode=fill_mode, **kwargs)
    
        params = np.linalg.inv(transform_matrix) if inv else transform_matrix
        params = transform_matrix_offset_center(params, x.shape[0], x.shape[1])
        
        affine_matrix = params[:2, :2]
        offset = params[:2, 2]
        
        x = apply_to_channel(x, ndi.interpolation.affine_transform, 
                             channel_axis=(-1 if x.ndim > 2 else None), 
                             args=(affine_matrix, offset,), 
                             kwargs=dict(order=1, mode=fill_mode, cval=cval))
        # x = skimage.transform.warp(x, params, mode=fill_mode, cval=cval)
    else:
        x = pad_image(x, size=output_shape, is_target=True, fill_mode=fill_mode, **kwargs)
    x = crop_image(x, size=output_shape, is_target=True)
    
    if horizontal_flip:
        x = x[:, ::-1, ...]
    
    if vertical_flip:
        x = x[::-1, ...]
    
    return x


class ImageDataset(object):
    """ Dataset class (modified from MaskRCNN Dataset)
        Inherit this class for different task
    """
    def __init__(self, source, classes, add_bg=True):
        """ source: a str for source
            class_info: [(class_id, class_name)] * N
        """
        # map image_id -> X, y, scc
        self.images = dict()
        self.class_info = pd.DataFrame(False, index=source, columns=classes, dtype=bool)
        if add_bg:
            self.class_info.insert(0, 'bg', True)
    
    def add_image(self, image_id, data, labels=None, source="", args=dict()):
        """ Add image information to Dataset.
            Basically, each input should have a data tuple (X, y). 
            X, y could be matrix, data or image/mask paths, etc.
            Then implement load_data to connect input with data generator
            For train/validation, X, y should be both provided. 
            For inference, y could be None.
            class_name and source are used to organize images.
        """
        assert image_id not in self.images, "image_id already exists"
        self.images[image_id] = {"data": data, "args": args, 
                                 'source': source, 'labels': labels}
        self.class_info.loc[source, labels] = True
    
    def add_label(self, label, label_id=None):
        """ Add a new label to the dataset. 
            WARNING: provide label_id will shift label_id for existing classes.
            label name won't changed. So keep label_id=None.
        """
        if label in self.class_info.columns:
            return
        if label_id is None:
            label_id = self.num_labels
        self.class_info.insert(label_id, label, False)
    
    def load_data(self, image_id):
        """ Parse data to data_generator.
            Apply processor in (__call__) to image in this function.
        """
        raise NotImplementedError("Subclass must implement this function")
    
    def __call__(self, **kwargs):
        """ Load data. """
        raise NotImplementedError("Subclass must implement this function")
    
    def get_labels_from_source(self, source):
        """ Get labels from a given source. 
            Return a boolean list x, len(x) = self.class_info.ncol.
        """
        return self.class_info.loc[source].tolist()
    
    def get_source_with_labels(self, labels):
        """ Get source contains a given label. 
            Return a boolean list x, len(x) = self.class_info.nrow.
        """
        return self.class_info.loc[:, labels].tolist()
    
    def items(self):
        """ An iterator for all images. """
        for k, v in self.images.items():
            yield k, v
    
    @property
    def image_ids(self):
        return list(self.images.keys())
    
    @property
    def num_images(self):
        return len(self.images)

    @property
    def num_labels(self):
        return len(self.class_info.columns)
    
    @property
    def image_info(self):
        """An alias of self.images used in matterplot mrcnn package."""
        return self.images
    
    def get_source_class_ids(self, source):
        """An alias of get_labels_from_source used in matterplot mrcnn package."""
        return self.get_labels_from_source(source)
    

class ImageDataProcessor(object):
    """ Rescale, Standardize, Augment the image
        Process the image with the following procedures.
        1. Rescale: rgb2gray, deconvolution, color dodge, equalization, etc.
        2. Standardize: maybe standardize images with mean and std.
        3. Padding: pad images before augmentation to keep original infos.
        4. Augment: maybe (randomly) augment images.
        5. Resize: resize images to model input size.
        
        Global args and function specific args could be used.
    """
    
    def __init__(self,
                 rescale=False,
                 standardize=False,
                 augment=False,
                 resize=False,
                 args=None):
        self.rescale = rescale
        self.standardize = standardize
        self.augment = augment
        self.resize = resize
        
        self.args = dict(mean=0.0, std=1.0, model_input_size=None, fill_mode='reflect', cval=0.)
        if args is not None:
            self.args.update(args)
        
        ## Default global mean/std with default value None
        #tensorflow_mean = np.array([123.68, 116.779, 103.939])
        #tensorflow_std = 1.0
        #torch_mean = [0.485, 0.456, 0.406]
        #torch_std = [0.225, 0.224, 0.229]


    def __call__(self, x, args=None):
        """
            # Arguments
            x: the input image.
            args:
            
            # Returns
            The image after treatment.
        """
        args = self.merge_args(x, args)

        if self.rescale:
            x = self.rescale_func(x, args)
        
        ## Get sample mean, std after rescale and before augment/resize
        if isinstance(args['mean'], str) and args['mean'] == 'sample_mean':
            args['mean'] = np.mean(x)
        if isinstance(args['std'], str) and args['std'] == 'sample_std':
            args['std'] = np.std(x) + 1e-8

        if self.augment:
            x = self.augment_func(x, args)
        
        if self.resize:
            x = self.resize_func(x, args)
        
        if self.standardize:
            x = self.standardize_func(x, args)

        x = self.output_image(x, args)
        
        return x

    def merge_args(self, x, img_args):
        """ Normalize the image by given meand and std. 
            Higher lvl mean/std will replace lower lvl mean/std
            self.args['mean/std'] < args['mean/std'] < samplewise['mean/std']
        """
        args = self.args.copy()
        args['image_size'] = x.shape[:-1]
        if img_args is not None:
            args.update(img_args)
        
        # If model_input_size is not provided, return original image size
        if args['model_input_size'] is None:
            args['model_input_size'] = args['image_size']
        
        return args
    
    def rescale_func(self, x, args):
        rescale_method = 'hist'
        if x.ndim < 3:
            return rescale_channel(x, rescale_method, **args)
        else:
            # rgb image do rescale on each channel
            r = rescale_channel(x[:,:,0], rescale_method, **args)
            g = rescale_channel(x[:,:,1], rescale_method, **args)
            b = rescale_channel(x[:,:,2], rescale_method, **args)
            return np.stack([r, g, b], axis=-1)
    
    def augment_func(self, x, args):
        """ Augment image by given transformation matrix and flipping info """
        # The length of diag was used to make sure image won't loose
        # information during augmentation

        transform_matrix = (args['transform_matrix']
                            if 'transform_matrix' in args else None)
        horizontal_flip = (args['horizontal_flip']
                           if 'horizontal_flip' in args else False)
        vertical_flip = (args['vertical_flip']
                         if 'vertical_flip' in args else False)
        inv = args['inv'] if 'inv' in args else False
        output_shape = (args['aug_output_shape'] 
                        if 'aug_output_shape' in args else None)
        
        return transform_image(x, output_shape = output_shape, 
                               transform_matrix = transform_matrix,
                               inv = inv, 
                               fill_mode = args['fill_mode'],
                               cval = args['cval'],
                               horizontal_flip = horizontal_flip,
                               vertical_flip = vertical_flip)
    
    def resize_func(self, x, args):
        return skimage.transform.resize(x, args['model_input_size'], 
                                        mode=args['fill_mode'], 
                                        cval=args['cval'])
    
    def standardize_func(self, x, args):
        return ((x - args['mean'])/args['std'])
    
    def output_image(self, x, args):
        return x

import os
import sys
import math

import keras.backend as K
import keras.preprocessing.image as KP_image
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io

from DIPModels.utils_g import utils_image
from DIPModels.utils_g import utils_keras
from DIPModels.utils_g import utils_misc

SEED = None

class TrainDataset(utils_image.ImageDataset):
    """ Generates the nuclei dataset.
        Argument: 
            processor: a DataImageProcessor object that preprocess the image.
            **kwargs: Future parameters
    """
    def __init__(self, source, classes, add_bg=True):
        super(TrainDataset, self).__init__(source, classes, add_bg)
        self.image_id_mapping = dict()
    
    def __call__(self, image_dir, processor, tf_args=None):
        """ Load training dataset and image processor.
            Argument:
                image_dir: the dir contains all trianing images.
                processor: a DataImageProcessor object that preprocess the image.
                tf_args: transformation parameters to augmentation function.
        """
        self.processor = processor
        tf_args = dict() if tf_args is None else tf_args.copy() # copy over any additional parameters if they exist
        for image_id in os.listdir(image_dir):
            # find the sample folders
            if os.path.isdir(os.path.join(image_dir, image_id)):
                # store the file paths of image and mask
                image_path = os.path.join(image_dir, image_id, "images", image_id + ".png")
                masks_path = os.path.join(image_dir, image_id, "masks")
                
                self.add_image(image_id=image_id, data=(image_path, masks_path), 
                               labels=['nuclei'], source="DSB_2018", args=tf_args)
                self.image_id_mapping[image_id] = len(self.image_id_mapping)
    
    def _update_args(self, args):
        ## Estimated number of masks under model_input_size
        model_input_size = self.processor.args['model_input_size']
        masks_rate = (model_input_size[0] * model_input_size[1]) / (args['image_size'][0] * args['image_size'][1])
        est_masks_per_image = args['num_masks'] * masks_rate
        ## Randomly pick a patch if there are too many masks in the image
        ## Parse a parameter to resize_func
        maximum_num_masks = self.processor.args['maximum_num_masks']
        if est_masks_per_image > maximum_num_masks:
            resize_rate = (maximum_num_masks/est_masks_per_image) ** 0.5
            est_h, est_w = math.floor(model_input_size[0] * resize_rate), math.floor(model_input_size[1] * resize_rate)
            init_h = np.random.randint(model_input_size[0] - est_h + 1)
            init_w = np.random.randint(model_input_size[1] - est_w + 1)
        else:
            init_h, init_w = 0, 0
            est_h, est_w = model_input_size
        ## Add the start, end location to to resize_func
        args['resize_loc'] = (init_h, init_w, init_h+est_h, init_w+est_w)

        transform_pars = utils_image.random_transform_pars(1, args)[0]
        args['transform_matrix'] = utils_image.generate_transform_matrix(transform_pars)
        return
        
    def load_data(self, image_id):
        """ Load image and masks. 
            image: (h, w, 3) numpy array.
            masks: (h, w, num_mask)
            label: (num_mask,)
        """
        ## Read in image and masks
        image = 1.0 * skimage.io.imread(self.images[image_id]['data'][0])[..., :3]
        masks = utils_misc.read_masks_from_dir(self.images[image_id]['data'][1], suffix='.png')
        args = dict(image_size = (image.shape[0], image.shape[1]), 
                    is_gray = np.all(image[:,:,0] == image[:,:,1]),
                    num_masks = masks.shape[-1])
        args.update(self.images[image_id]['args'])
        self._update_args(args)
        
        args['is_mask'] = False
        image = self.processor(image, args)
        args['is_mask'] = True
        masks = self.processor(masks, args)
        class_ids = np.ones(masks.shape[-1], dtype=np.int32)
            
        return image, (masks.astype(bool), class_ids)

class ValidDataset(TrainDataset):
    def _update_args(self, args):
        ## Estimated number of masks under model_input_size
        model_input_size = self.processor.args['model_input_size']
        est_h, est_w = min(args['image_size'][0], model_input_size[0]), min(args['image_size'][1], model_input_size[1])
        init_h = np.random.randint(args['image_size'][0] - est_h + 1)
        init_w = np.random.randint(args['image_size'][1] - est_w + 1)
        args['resize_loc'] = (init_h, init_w, init_h+est_h, init_w+est_w)
        args['transform_matrix'] = None
        return

class InferenceDataset(ValidDataset):
    def __call__(self, image_dir, processor, tf_args):
        """ Load training dataset and image processor.
            Argument:
                image_dir: the dir contains all trianing images.
                processor: a DataImageProcessor object that preprocess the image.
                tf_args: transformation parameters to augmentation function.
        """
        self.processor = processor
        tf_pars = utils_image.product_transform_pars(tf_args)
        tf_matrix = [utils_image.generate_transform_matrix(x) for x in tf_pars]
        for image_id in os.listdir(image_dir):
            if os.path.isdir(os.path.join(image_dir, image_id)):
                image_path = os.path.join(image_dir, image_id, "images", image_id + ".png")
                
                self.add_image(image_id=image_id, data=(image_path, None), 
                               labels=['nuclei'], source="DSB_2018", args=tf_matrix)
                self.image_id_mapping[image_id] = len(self.image_id_mapping)    
    
    def load_data(self, image_id):
        ## Read in image and masks
        image = 1.0 * skimage.io.imread(self.images[image_id]['data'][0])[..., :3]
        image_size = (image.shape[0], image.shape[1])
        is_gray = np.all(image[:,:,0] == image[:,:,1])
        res = []
        for i, tf_matrix in enumerate(self.images[image_id]['args']):
            args = dict(image_size=image_size, is_gray=is_gray, is_mask=False)
            self._update_args(args)
            args['transform_matrix'] = tf_matrix
            res.append(self.processor(image, args))
        return res

    def inv_transform(self, x, image_id, is_mask):
        args = self.processor.merge_args(x, self.image_info[image_id]['args'])
        args['inv'] = True
        args['aug_output_shape'] = x.shape[:2]
        # args['aug_output_shape'] = args['image_size']
        x = self.processor.augment_func(x, args)
        
        if is_mask:
            x = np.where(x > 0, 1.0, 0.0)
        check = np.logical_and(np.where(x != 0, True, False), np.where(x != 1, True, False))
        print(np.sum(check))
        return x
        # return utils_image.crop_image(x, args['image_size'], is_target=True)
    
class TrainProcessor(utils_image.ImageDataProcessor):
    """ Data Processor for training image preprocessing """
    def rescale_func(self, x, args):
        """ Output range 0.0~1.0 """
        if args['is_mask']:
            return x
        else:
            x = x * 1.0
            rescale_method = 'hist'
            if x.ndim < 3:
                return utils_image.rescale_channel(x, rescale_method, **args)
            else:
                # rgb image do rescale on each channel
                # return utils_image.apply_to_channel(x, utils_image.rescale_channel, kwargs=dict(rescale_method=rescale_method, **args)
                r = utils_image.rescale_channel(x[:,:,0], rescale_method, **args)
                g = utils_image.rescale_channel(x[:,:,1], rescale_method, **args)
                b = utils_image.rescale_channel(x[:,:,2], rescale_method, **args)
                return np.stack([r, g, b], axis=-1)
                # return utils_image.rescale_intensity_with_deconv(x, out_range=(0, 1), **args)
    
    def augment_func(self, x, args):
        """ (1) pad image with border.
            (2) augment image and then resize to model_input_size.
        """
        edge_pad_size = args['edge_pad_size']
        edge_fill_mode = 'constant' if args['is_mask'] else 'linear_ramp'

        x = utils_image.pad_image(x, size=edge_pad_size, is_target=False, fill_mode=edge_fill_mode)
        args['aug_output_shape'] = args['model_input_size']
        x = super(TrainProcessor, self).augment_func(x, args)
        return x
    
    def resize_func(self, x, args):
        """ (1) If estimated_num_masks > 'maximum_num_masks', 
                Randomly pick a patch contains less than 
                'maximum_num_masks' masks. And resize the patch. 
            (2) Reorganize masks
        """
        init_h, init_w, end_h, end_w = args['resize_loc']
        x = x[init_h:end_h, init_w:end_w, :]
        x = utils_image.pad_image(x, size=args['model_input_size'], is_target=True, 
                                  fill_mode='constant', constant_values = args['cval'])
        return x
    
    def standardize_func(self, x, args):
        if args['is_mask']:
            return x
        return ((x - args['mean'])/args['std'])
    
    def output_image(self, x, args):
        # separate masks on same layers
        if args['is_mask']:
            masks_list = [utils_misc.image_to_masks(x[:,:,i])[0]
                          for i in range(x.shape[-1])]
            x = np.concatenate(masks_list, axis=-1)
            return x
        else:
            return x
    
    
class InferenceProcessor(TrainProcessor):
    def augment_func(self, x, args):
        args['aug_output_shape'] = args['model_input_size']
        x = super(TrainProcessor, self).augment_func(x, args)
        return(x)


## Malisiewicz et al.
def non_max_suppression_boxes(rois, masks, class_ids, scores, threshold):
    new_rois, new_masks, new_class_ids, new_scores = [], [], [], []
    
    # grab the coordinates of the bounding boxes
    x1 = rois[:,0]
    y1 = rois[:,1]
    x2 = rois[:,2]
    y2 = rois[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list
        i = idxs[-1]
        
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])
        
        # compute the IoU scores
        intersect = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        IoU = (1.0 * intersect) / (area[idxs] + area[i] - intersect + 1e-8)
        
        # Find IoU scores > threshold and remove low score masks
        pick = np.where(np.logical_and(IoU > threshold,
                                       class_ids[idxs] == class_ids[i]))[0]
        im = idxs[pick]
        im = im[np.where(scores[im] >= scores[im].max() - 0.2)]
        
        # merge boxes, masks, scores
        new_rois.append([rois[im, 0].min(), rois[im, 1].min(),
                         rois[im, 2].max(), rois[im, 3].max()])
        new_masks.append(np.any(masks[:, :, im], axis=-1))
        new_scores.append(np.mean(scores[im]))
        new_class_ids.append(class_ids[i])
        
        idxs = np.delete(idxs, pick)
    return dict(rois=np.array(new_rois),
                masks=np.rollaxis(np.stack(new_masks), 0, 3),
                class_ids=np.array(new_class_ids),
                scores=np.array(new_scores))

# def non_max_suppression_masks(rois, masks, class_ids, scores, threshold):
def non_max_suppression_masks(masks, scores, threshold=0.7):
    assert len(scores) == masks.shape[2], "Masks and scores do not match"
    N = len(scores)
    idxs = np.argsort(-scores)
    masks, scores = masks[:, :, idxs], scores[idxs]
    # rois, class_ids = rois[idxs,:], class_ids[idxs]
    keep = [True] * N
    for i in range(N):
        if not keep[i]:
            continue
        for j in range(i + 1, N):
            intersect = np.sum(np.logical_and(masks[:, :, i], masks[:, :, j]))
            union = np.sum(np.logical_or(masks[:, :, i], masks[:, :, j]))
            if intersect/union > threshold:
                keep[j] = False
    
    # return dict(rois=rois[keep, :], masks=masks[:, :, keep],
    #             class_ids=class_ids[keep], scores=scores[keep])
    return masks[:, :, keep], scores[keep]

def remove_overlapping_masks(masks, scores, threshold=0.7):
    h, w, d = masks.shape
    assert len(scores) == d, "Masks and scores should have same length"
    idxs = np.argsort(-scores)
    masks, scores = masks[:, :, idxs], scores[idxs]
    
    occlusion = np.ones((h, w))
    toKeep = [False for i in range(d)]
    for i in range(d):
        # Remove masks overlap with previously detected region
        originalArea = np.sum(masks[:, :, i])
        masks[:, :, i] = masks[:, :, i] * occlusion
        newArea = np.sum(masks[:, :, i])
        if newArea > 0 and newArea/originalArea > threshold:
            toKeep[i] = True
            occlusion = np.logical_and(occlusion, np.logical_not(masks[:,:,i]))
    return masks[:, :, toKeep], scores[toKeep]

def remove_outlier_masks(masks, scores, quantile=np.array([0.1, 10])):
    # Remove very large/small masks
    areas = np.sum(masks, axis=(0, 1))
    lb, ub = np.median(areas[areas > 0]) * quantile
    toKeep = (areas > lb) & (areas < ub) & (scores > 0.5)
    return masks[:, :, toKeep], scores[toKeep], (lb, ub)

def remove_low_score_masks(masks, scores, x):
    score_map = get_score_map(masks, scores, intensity=x)
    mean_mask_scores = np.array([np.sum(masks[:,:,i] * score_map)/np.sum(masks[:,:,i]) 
                        for i in range(masks.shape[2])])
    print(np.percentile(mean_mask_scores, np.arange(0, 1, 0.1)))
    toKeep = mean_mask_scores > 0.005
    return masks[:, :, toKeep], scores[toKeep]

def get_score_map_Unet(x, model):
    ## TODO: use Unet to generate scores map on top of layer[360]
    ### Build an extra UNet on top of this if have time
    #score_model = keras.Model(inputs=[model.keras_model.inputs[0]], outputs=[model.keras_model.layers[360].output])
    # score_model.summary()
    #RCNN_layer_weights = dict([(layer.name, layer) for layer in model.keras_model.layers])
    #for layer in score_model.layers:
    #    layer.set_weights(RCNN_layer_weights[layer.name].get_weights())
    #molded_images, image_metas, windows = model.mold_inputs([image_preprocessed])
    #res = score_model.predict(molded_images)
    pass

def get_score_map(masks, scores, intensity=None):
    score_map = [np.where(masks[:, :, i], score, 1-score)
                 for i, score in enumerate(scores)]
    score_map = np.mean(np.stack(score_map, axis=-1), axis=-1)
    if intensity is not None:
        score_map = score_map * skimage.exposure.rescale_intensity(intensity, out_range=(0, 1))
    return score_map

def merge_results(x, results, transform_matrix, output_size=None, pad_image=False, plot=False, eps=1e-3):
    ## smaller eps will result in smaller inner region and larger edge region
    x = skimage.color.rgb2gray(x)*1.0
    output_size = x.shape[:2] if output_size is None else output_size
    
    ## Merge masks and scores from results with different transformation
    masks = [utils_image.transform_image(r['masks'], tf_matrix, inv=True, output_shape=None, 
                                         fill_mode='constant', cval=0.)
             for r, tf_matrix in zip(results, transform_matrix) if len(r['scores']) > 0]
    masks = np.concatenate([r['masks'] for r in results], axis=-1)
    scores = np.hstack([r['scores'] for r in results])
    # Pad raw image to mask size
    x = utils_image.pad_image(x, masks.shape[:2], is_target=True)
    
    # Remove outlier and low score masks
    masks, scores, (lb, ub) = remove_outlier_masks(masks, scores, quantile=np.array([0.1, 10]))
    print("after remove outlier")
    plt.imshow(utils_misc.masks_to_image(masks, labels='all'))
    plt.show()
    masks, scores = remove_low_score_masks(masks, scores, x)
    print("after remove low score")
    plt.imshow(utils_misc.masks_to_image(masks, labels='all'))
    plt.show()
    
    # Get score_map and use the gradients as watershed image
    score_map = get_score_map(masks, scores, intensity=x)
    if pad_image:
        score_map = utils_image.pad_image(score_map, [(30, 30), (30, 30)])
    # gradients = skimage.filters.rank.gradient(score_map_pad, skimage.morphology.disk(1))
    gradients = skimage.filters.scharr(score_map)
    gradients = np.where(gradients > eps, gradients, 0)
    
    # Remove redundant masks
    masks, scores = non_max_suppression_masks(masks, scores, threshold=0.7)
    masks, scores = remove_overlapping_masks(masks, scores, threshold=0.7)
    print("after remove over lapping")
    plt.imshow(utils_misc.masks_to_image(masks, labels='all'))
    plt.show()
    
    # Reduce the size of masks and use as watershed marker
    fg = utils_misc.masks_to_image(masks, labels='all')
    if pad_image:
        fg = utils_image.pad_image(fg, [(30, 30), (30, 30)])
    bg = np.where(fg + gradients > 0, False, True)
    markers = (skimage.morphology.erosion(np.where(fg > 0, fg + 1, 0), skimage.morphology.disk(2))
               + skimage.morphology.erosion(bg, skimage.morphology.disk(4)))
    
    # Watershed segmentation for region
    region_labels = skimage.morphology.watershed(gradients, markers) - 1
    # region_labels = skimage.morphology.closing(region_labels)
    if plot:
        print("Watershed for nuclei")
        utils_misc.plot_watershed(x, markers, gradients, region_labels)
    
    # Remove outlier masks
    #masks = utils_misc.image_to_masks(region_labels, criteria=lambda x: x.area > lb and x.area < ub)[0]
    # masks, scores = remove_outlier_masks(masks, scores, quantile=np.array([0.1, 10]))
    #region_labels = utils_misc.image_to_labels(masks, labels='all')
    
    # Watershed segmentation for edge
    edge_markers = skimage.morphology.dilation(region_labels, skimage.morphology.disk(4))
    edge_labels = np.where(gradients > 0, edge_markers, 0)
    if plot:
        print("Nuclei-edge")
        utils_misc.plot_watershed(gradients, edge_markers, region_labels, edge_labels)
    
    # Merge edge and region and shrink result to input image_size
    labels = np.where(region_labels > 0, region_labels, edge_labels)
    # labels = region_labels
    utils_misc.plot_watershed(x, labels, region_labels, edge_labels)
    
    labels = utils_image.crop_image(labels, output_size, is_target=True)
    markers = utils_image.crop_image(markers, output_size, is_target=True)
    gradients = utils_image.crop_image(gradients, output_size, is_target=True)
    score_map = utils_image.crop_image(score_map, output_size, is_target=True)
    
    return markers, gradients, labels, score_map

## TODO: Add batch_size
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

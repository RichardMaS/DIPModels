import csv
import os
#import xlrd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import skimage.feature
import skimage.measure
import skimage.morphology
import skimage.io


#### TODO: Remove this to specific project or rewrite ####
def load_images_from_dir(data_dir):
    for image_id in os.listdir(data_dir):
        sub_folder = os.path.join(data_dir, image_id)
        if os.path.isdir(sub_folder):
            image, masks = None, None
            if "images" in os.listdir(sub_folder):
                # image = cv2.imread(os.path.join(sub_folder,
                # "images", file_id + ".png"))[:,:,(2,1,0)]
                image = skimage.io.imread(os.path.join(sub_folder, "images", image_id + ".png"))[..., :3]
            if "masks" in os.listdir(sub_folder):
                masks = read_masks_from_dir(os.path.join(sub_folder, "masks"))
            yield image, masks, image_id

def read_masks_from_dir(mask_dir, suffix='.png'):
    """ Read all mask files in a mask_dir. 
        Return a np.array with shape [height, width, N_masks]
    """
    mask_list = []
    for file in os.listdir(mask_dir):
        if file.endswith(suffix):
            mask = skimage.io.imread(os.path.join(mask_dir, file))
            mask = skimage.img_as_float(mask)
            mask_list.append(mask)
    return np.stack(mask_list, axis=-1).astype(np.uint8)

#def excel_to_csv(excel_file, csv_file):
#    """ Transfer excel file to csv file """
#    wb = xlrd.open_workbook(excel_file)
#    sh = wb.sheet_by_name('Sheet1')
#    out_file = open(csv_file, 'w')
#    wr = csv.writer(out_file, quoting=csv.QUOTE_ALL)

#    for rownum in range(sh.nrows):
#        wr.writerow(sh.row_values(rownum))

#    out_file.close()

#### TODO: Rewrite this function ####
def write_masks_to_file(masks_dict, filename):
    """ Write masks into files with rle_encoding
    
    Augment:
        masks_dict: a dictionary with ImageId as key and masks matrix as value
        filename: output filename
    """
    with open(filename, "w") as myfile:
        myfile.write("ImageId, EncodedPixels\n")
        for image_id, masks in masks_dict.iteritems():
            print(image_id)
            RLE = rle_encoding(masks[:, :, j])[0]
            myfile.write(image_id + "," + " ".join([str(k) for k in RLE]) + "\n")

def get_masks_from_rles(df_rles, image_id, h, w, fill_value=1):
    if isinstance(df_rles, str):
        df_rles = pd.read_csv(df_rles, index_col="ImageId")
    res = []
    for rle in df_rles.loc[image_id, "EncodedPixels"].tolist():
        res.append([int(x) for x in rle.split(' ')])
    print(len(res))
    return run_length_decode(res, h, w, fill_value)
    
def masks_to_image(masks, labels=None):
    """ Transfer masks(h, w, N) + labels(N,) into image. 
        Default labels=None: all masks will be assigned 
        with value 1. labels='all': will set each mask
        with a unique label.
    """
    if masks is None:
        return masks
    if masks.ndim == 2:
        np.expand_dims(masks, axis=-1)
    if labels is None:
        labels = np.ones(masks.shape[-1])
    elif labels == "all":
        labels = np.arange(masks.shape[-1])+1
    return np.dot(masks, labels)

def image_to_masks(x, criteria=lambda x: True):
    """ Transfer image back to masks (h, w, N) + labels(N,). """
    mask_list = [np.zeros(x.shape + (0,), dtype=np.uint8)]
    labels = list()
    for prop in skimage.measure.regionprops(skimage.measure.label(x), intensity_image=x):
        if criteria(prop):
            mask = np.zeros(x.shape + (1,), dtype=np.uint8)
            mask[prop.coords[:,0], prop.coords[:,1], ...] = 1
            mask_list.append(mask)
            labels.append(np.ceil(prop.mean_intensity))
    return np.concatenate(mask_list, axis=-1), np.array(labels, dtype=np.uint8)

def run_length_encode(masks):
    h, w, d = masks.shape
    res = []
    for i in range(d):
        bs = np.where(masks[:,:,i].T.flatten())[0] + i * h * w
        
        rle = []
        prev = -2
        for b in bs:
            if (b > prev + 1):
                rle.extend((b + 1, 0))
            rle[-1] += 1
            prev = b

        #https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
        #if len(rle)!=0 and rle[-1]+rle[-2] == x.size:
        #    rle[-2] = rle[-2] -1  #print('xxx')
        res.append(rle)
    # rle = ' '.join([str(r) for r in rle])
    return res

def run_length_decode(rles, h, w, fill_value=1):
    res = []
    # rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for x in rles:
        mask = np.zeros((h * w), np.uint8)
        rle = np.array(x).reshape(-1, 2)
        for r in rle:
            start = (r[0] - 1) % (w * h)
            end = start + r[1]
            mask[start : end] = fill_value
        res.append(mask.reshape(w, h).T)
    
    return np.stack(res, axis=-1)

## Function to calculate IoU scores
def get_iou(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    nb_true, nb_pred = len(np.unique(y_true)), len(np.unique(y_pred))
    print("True-Class vs. Predicted-Class:", [nb_true-1, nb_pred-1])
    
    # Calculate A_and_B, A_or_B and iou_score
    A_and_B = np.histogram2d(y_true, y_pred, bins=(nb_true, nb_pred))[0]
    A_or_B = (np.histogram(y_true, bins = nb_true)[0].reshape((-1, 1))
              + np.histogram(y_pred, bins = nb_pred)[0]
              - A_and_B) + 1e-8
    return (A_and_B / A_or_B)[1:, 1:]

def get_confusion_matrix(iou_scores, threshold):
    matches = iou_scores > threshold
    true, pred = np.any(matches, axis=1), np.any(matches, axis=0)
    ## mat = [[tp, fn], [(tp), fp]] precision = mat[0,0]/(sum(mat)-mat[0,0])
    matrix = [[np.where(true)[0]+1, np.where(np.logical_not(true))[0]+1], 
              [np.where(pred)[0]+1, np.where(np.logical_not(pred))[0]+1]]
    tp, fn, tn, fp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    p = len(tp) / (len(tp) + len(fp) + len(fn))
    # print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(threshold, len(tp), len(fp), len(fn), p))
    return p, matrix

def analyze_masks_error(matrix, y_true, y_pred):
    tp, fn, tn, fp = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    base_img = np.zeros(y_true.shape)
    img_predicted = base_img + sum([np.where(y_true==x, 50, 0) for x in tp] + [np.where(y_pred==x, 100, 0) for x in tn])
    img_unpredicted = base_img + sum([np.where(y_true==x, 50, 0) for x in fn] + [np.where(y_pred==x, 100, 0) for x in fp])
    return img_predicted, img_unpredicted

def display_np_array(x):
    return [x.shape, x.dtype] + ([x.max(), x.min()] if min(x.shape) > 0 else [None, None])

def plot_images(images, titles=None, cmaps=None, print_stats=True, **kwargs):
    assert len(images) == len(titles)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * len(titles)
    nb_plot = len([x for x in images if x is not None])
    fig, axes = plt.subplots(1, nb_plot, figsize=[4 * nb_plot, 4],
                             sharex=False, sharey=False)
    ax = axes.ravel()
    k = 0
    for img, title, cmap in zip(images, titles, cmaps):
        if img is not None:
            if title:
                ax[k].set_title(title)
            ax[k].imshow(img, cmap=cmap, **kwargs)
            if print_stats:
                print([title, img.shape, img.dtype, img.max(), img.min()])
            k += 1
    plt.show()

def plot_image_masks(image_raw=None, masks_raw=None, image_processed=None,
                     masks_processed=None, masks_predict=None):
    if image_raw is not None:
        print('Raw Image: %s' % display_np_array(image_raw))
        image_raw = image_raw.astype(np.uint8)
    if masks_raw is not None:
        print('Raw Masks: %s' % display_np_array(masks_raw))
        masks_raw = masks_to_image(masks_raw, labels='all')
    if image_processed is not None:
        print('Processed Image: %s' % display_np_array(image_processed))
        image_processed = skimage.exposure.rescale_intensity(image_processed, out_range=(0, 1))
    if masks_processed is not None:
        print('Processed Masks: %s' % display_np_array(masks_processed))
        masks_processed = masks_to_image(masks_processed, labels='all')
    if masks_predict is not None:
        print('Predicted Masks: %s' % display_np_array(masks_predict)) 
        masks_predict = masks_to_image(masks_predict, labels='all')

    images = [image_raw, masks_raw, image_processed,
              masks_processed, masks_predict]
    titles = ["Raw Image", "Raw Masks", "Processed Image",
              "Processed Masks", "Predicted Masks"]
    cmaps = [None, plt.cm.nipy_spectral, plt.cm.gray,
             plt.cm.nipy_spectral, plt.cm.nipy_spectral]
    plot_images(images=images, titles=titles,
                cmaps=cmaps, interpolation='nearest', print_stats=False)

def plot_watershed(image=None, markers=None, gradients=None, labels=None):
    if image is not None:
        image = skimage.exposure.rescale_intensity(image, out_range=(0, 1))
    images = [image, markers, gradients, labels]
    titles = ["Grayscale Image", "Markers", "Gradients", "Watershed Labels"]
    cmaps = [plt.cm.gray, plt.cm.nipy_spectral,
             plt.cm.nipy_spectral, plt.cm.nipy_spectral]
    plot_images(images=images, titles=titles,
                cmaps=cmaps, interpolation='nearest')

def config_cmd_parser(config):
    members = [attr for attr in dir(config) 
               if not callable(getattr(config, attr)) and not attr.startswith("__")]
    return members   


import numpy as np
import cv2, sys
from skimage.feature import hog
from scipy.ndimage.measurements import label
from lib import np_util as npu
from lib.helpers import _x0,_x1,_y0,_y1


def get_hog(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False):
    ''' Ease calling of hog() by generating tuples from single value for 2 parameters.
    NOTE: hog returns a single value if vis=False, but tuple if vis=True
    feature_vecture=True will return data as feature vector by calling
     .ravel() on result.
    '''
    return hog(img, orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    transform_sqrt=True,
                    visualise=vis, feature_vector=feature_vec)

def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()

def color_hist(img, nbins=32, bins_range=(0, 256)):
    ''' Color histogram - Compute histogram of the color channels separately,
     then concatenate them into a single feature vector
    bins_range: NEEDS CHANGE if reading .png files with mpimg!
    '''
    ch1 = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ch2 = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    ch3 = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    return np.concatenate((ch1[0], ch2[0], ch3[0]))

def hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
    vis=False, feature_vec=True):
    ''' Histogram of Gradients features
    Returns hog() features one or all channels 
    If "ValueError: operands could not be broadcast together with shapes...",
     it is likely get_hog() returns ravel() when it should for all channels.
     So get_hog() is called with feature_vec=False and ravel() is only call
     on the result if all channels.
    For hog that is to be subsampled, this fn needs to be called with
     feature_vec=False so subsamples can call ravel()
    '''
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            hog_features.append(get_hog(img[:,:,channel], orient, 
                                        pix_per_cell, cell_per_block,
                                        vis=vis, feature_vec=False))
        feats = np.array(hog_features)
        return feats.ravel() if feature_vec else feats
    else:
        return get_hog(img[:,:,hog_channel], orient,
                                        pix_per_cell, cell_per_block, 
                                        vis=vis, feature_vec=feature_vec)

def image_features(img, color_space=None, spatial_size=(32, 32), hist_bins=32, 
                   orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                   spatial_feat=True, hist_feat=True, hog_feat=True, 
                   hog_feats=None, concat=False, dbg=False):
    ''' Extract features of an image
    '''
    out = []
    img = npu.RGBto(color_space, img)

    if spatial_feat:
        out.append(bin_spatial(img, size=spatial_size))
    if hist_feat:
        out.append(color_hist(img, nbins=hist_bins))
    if hog_feats!=None:
        out.append(hog_feats)
    elif hog_feat:
        out.append(hog_features(img, orient, pix_per_cell, cell_per_block, hog_channel))
    return np.concatenate(out) if concat else out

def images_features(imgs, color_space='RGB', spatial_size=(32, 32),
                    hist_bins=32, orient=9,
                    pix_per_cell=8, cell_per_block=2, hog_channel=0,
                    spatial_feat=True, hist_feat=True, hog_feat=True):
    ''' Extract features from a list of images
    '''
    result = []
    for file in imgs:
        # Read in each one by one
        img = npu.BGRto('RGB', cv2.imread(file))
        features = image_features(img, color_space, spatial_size, hist_bins,
            orient, pix_per_cell, cell_per_block, hog_channel, 
            spatial_feat, hist_feat, hog_feat)
        result.append(np.concatenate(features))
    return result

def horizontal_bboxes(win_w, step, y, xmax, xmin=0, win_h=None):
    ''' Returns list of bounding box coords that increments by step pixels horizontally
    win_h: set to win_w if None
    step: % of win_w
    '''
    result = []
    x = xmin
    y1 = y + win_w if win_h==None else y + win_h
    while int(x)+win_w <= xmax:
        result.append(((int(x), y), (int(x)+win_w, y1)))
        x += step*win_w
    return result

def next_width(y1, ht, ybase=440, top_w=32, btm_w=360):
# def next_width(y1, ht, ybase=420, top_w=32, btm_w=360):
    ''' Get next width for horizontal bboxes based on perspectives
    '''
    y_ratio = (y1-ybase)/(ht-ybase)
    # print(y1, ht, y_ratio,btm_w , top_w, top_w, int(y_ratio*(btm_w - top_w) + top_w))
    return int(y_ratio*(btm_w - top_w) + top_w)

def sliding_box_rows(img_shape, ymin=360, ymax=None, max_h=320, 
# def sliding_box_rows(img_shape, ymin=360, ymax=None, max_h=280, 
# def sliding_box_rows(img_shape, ymin=None, ymax=None, max_h=.5, 
    # xstep=.1, ystep=.2, min_w=64, dbg=False):
    xstep=.05, ystep=.2, min_w=64, dbg=False):
    ''' Returns rows of bounding box coords by sliding different size of windows
    for each row.
    Application is for vehicle detection, thus smaller windows row is near middle
    of image and no rows of same size is repeated.

    ymin: windows y start
    ymax: None = image ht
    xstep, ystep: % of win_w
    max_h: max window ht in % of imght if <= 1, in pxs otherwise
    min_w: min window wd in pxs
    '''
    img_h, img_w = img_shape[:2]
    max_w = int(max_h*img_h) if 0<=max_h<=1 else int(max_h) 
    ymin = ymin if ymin!=None else img_h - max_w
    # ymax = ymax or img_h
    win_w = max_w
    y = ymin
    y1 = ymin + win_w
    rows = []

    while (win_w >= min_w):
        row = horizontal_bboxes(win_w, xstep, y, img_w)
        # if dbg: print('wd:', win_w, 'len', len(row), 'y', y)

        rows.append(row)
        y1 -= int(win_w * ystep)
        win_w = next_width(y1, img_h)
        y = y1 - win_w

    # if dbg:
    #     by_wds = np.array(strips_shifts).T
    #     for by_wd in by_wds:
    #         win0 = by_wd[0][0]
    #         print('\nwidth %d:' % win0[1][0])
    #         for s in by_wd:
    #             for win in s:
    #                 print(win)
    #     colors = [
    #      (0,255,0),
    #      (0,215,0),
    #      (0,175,0),
    #      (0,135,0),
    #      (0,100,0),
    #      (0,70,0),
    #      (0,40,0),
    #      (0,10,0),
    #     ]
    #     for i,shift in enumerate(strips_shifts):
    #         draw_image = np.zeros(img_shape)
    #         # print(len(shift),'\n',i)
    #         for j,stripe in enumerate(shift):
    #             draw_image = draw_boxes(draw_image, stripe, colors[j])
    #         # cv2.imshow('shifted_collection', draw_image)
    #         cv2.imwrite('output_images/slide_windows%d.jpg'%i, draw_image)
    return rows

def ymin_ymax(rows):
    ymin = sys.maxsize
    ymax = 0
    for row in rows:
        y0 = _y0(row[0])
        y1 = _y1(row[0])
        if y0 < ymin:
            ymin = y0
        if y1 > ymax:
            ymax = y1
        y0 = _y0(row[-1])
        y1 = _y1(row[-1])
        if y0 < ymin:
            ymin = y0
        if y1 > ymax:
            ymax = y1
    return ymin, ymax

def bboxes_of_heat(heatmap, threshold):
    ''' Returns bounding boxes of heat areas in heatmap image.
    '''
    filtered_heatmap = npu.threshold(heatmap, threshold)
    labelsAry, nfeatures = label(filtered_heatmap)
    bboxes = []
    for i in range(1, nfeatures+1):
        nonzero = (labelsAry==i).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bboxes.append(((np.min(nonzerox)-1, np.min(nonzeroy)-1),
                       (np.max(nonzerox)+1, np.max(nonzeroy)+1)))
    return bboxes

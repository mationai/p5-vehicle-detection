import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
from lib import np_util as npu


# Define a function to return HOG features and visualization
def get_hog(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    ''' Calls hog() by generating tuples for 2 of hog params
    NOTE: hog returns a single value if vis=False, but tuple if vis=True
    '''
    return hog(img, orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    transform_sqrt=True,
                    visualise=vis, feature_vector=feature_vec)

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
    vis=False, feature_vec=True):
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            hog_features.extend(get_hog(img[:,:,channel], orient, 
                                        pix_per_cell, cell_per_block,
                                        vis=vis, feature_vec=feature_vec))
    else:
        hog_features = get_hog(img[:,:,hog_channel], orient,
                                        pix_per_cell, cell_per_block, 
                                        vis=vis, feature_vec=feature_vec)
    return hog_features

def image_features(img, color_space=None, spatial_size=(32, 32),
                   hist_bins=32, orient=9,
                   pix_per_cell=8, cell_per_block=2, hog_channel=0,
                   spatial_feat=True, hist_feat=True, hog_feat=True):
    ''' Extract features of an image
    '''
    out = []
    img = npu.RGBto(color_space, img)

    if spatial_feat:
        out.append(bin_spatial(img, size=spatial_size))
    if hist_feat:
        out.append(color_hist(img, nbins=hist_bins))
    if hog_feat:
        out.append(hog_features(img, orient, pix_per_cell, cell_per_block, hog_channel))
    return out

def images_features(imgs, color_space='RGB', spatial_size=(32, 32),
                    hist_bins=32, orient=9,
                    pix_per_cell=8, cell_per_block=2, hog_channel=0,
                    spatial_feat=True, hist_feat=True, hog_feat=True):
    ''' Extract features from a list of images
    '''
    features = []
    for file in imgs:
        # Read in each one by one
        img = npu.BGRto(color_space, cv2.imread(file))
        file_features = image_features(img, color_space, spatial_size, hist_bins,
            orient, pix_per_cell, cell_per_block, hog_channel, 
            spatial_feat, hist_feat, hog_feat)
        features.append(np.concatenate(file_features))
    return features

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

def sliding_box_rows(img_shape, ymin=360, ymax=None, max_h=280, 
# def sliding_box_rows(img_shape, ymin=None, ymax=None, max_h=.5, 
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

def get_bboxes(heatmap, threshold):
    ''' Returns bounding boxes of heat areas in heatmap image.
    '''
    heat_thresh = npu.threshold(heatmap, threshold)
    labelsAry, nfeatures = label(heat_thresh)
    bboxes = []
    for i in range(1, nfeatures+1):
        nonzero = (labelsAry==i).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bboxes.append(((np.min(nonzerox)-1, np.min(nonzeroy)-1),
                       (np.max(nonzerox)+1, np.max(nonzeroy)+1)))
    return bboxes

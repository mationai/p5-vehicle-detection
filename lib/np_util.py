import numpy as np
import cv2
from types import SimpleNamespace as SNS
from skimage import feature as skFeat

greens = [
    (0,255,0),
    (0,210,0),
    (0,175,0),
    (0,150,0),
    (0,125,0),
    (0,100,0),
    (0,80,0),
    (0,60,0),
    (0,45,0),
    (0,30,0),
    (0,0,0),
]

def scale_255(M):
    return np.uint8(255*M/np.max(M))

def threshold(M, thres):
    M[M <= thres] = 0
    return M 

def BGRto(cs, img):
    # matplotlib.image.imread returns RGB
    if cs=='RGB': return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cs=='LUV': return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    if cs=='YUV': return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if cs=='HSV': return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if cs=='HLS': return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if cs=='YCrCb': return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img

def RGBto(cs, img):
    # cv2.imread returns BGR
    if cs=='BGR': return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if cs=='LUV': return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if cs=='YUV': return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if cs=='HSV': return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if cs=='HLS': return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if cs=='YCrCb': return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return img

def colorSpaceRanges(cs):
    return [(0,192),(0,256),(0,256)] if cs=='LUV' else [(0,256)]*3

def spatial_features(img, size=(32,32)):
    return cv2.resize(img, size).ravel()

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, bins=32, ranges=[(0,256)]*3):
    ch1 = np.histogram(img[:, :, 0], bins=bins, range=ranges[0])
    ch2 = np.histogram(img[:, :, 1], bins=bins, range=ranges[1])
    ch3 = np.histogram(img[:, :, 2], bins=bins, range=ranges[2])
    # ch1 = np.histogram(img[:, :, 0], bins=nbins)
    # ch2 = np.histogram(img[:, :, 1], bins=nbins)
    # ch3 = np.histogram(img[:, :, 2], bins=nbins)
    chs = [ch1, ch2, ch3]
    return SNS(
        hist=np.concatenate([ch[0] for ch in chs]),
        bin_edges=np.concatenate([ch[1] for ch in chs]),
    )

def hog(img,
    orientations=8,
    pxs_per_cell=8,
    cells_per_blk=2,
    channels=None, # 'all' or (0,1,2) for all
    visualise=False,
    feature_vector=False,
    returnList=False):
    ''' Histogram of Oriented Gradients Features channels
    '''
    if channels==None or len(channels)==1: 
        return skFeat.hog(img if channels==None else img[:,:,channels[0]], 
            orientations=orientations,
            pixels_per_cell=(pxs_per_cell,pxs_per_cell),
            cells_per_block=(cells_per_blk,cells_per_blk),
            transform_sqrt=True, visualise=visualise, feature_vector=feature_vector
        )
    hog_features = []
    chs = (0,1,2) if channels.lower()=='all' else channels
    for ch in chs:
        hog_features.append(skFeat.hog(img[:,:,ch], 
            orientations=orientations,
            pixels_per_cell=(pxs_per_cell,pxs_per_cell),
            cells_per_block=(cells_per_blk,cells_per_blk),
            transform_sqrt=True, visualise=visualise, feature_vector=feature_vector
        ))
    return hog_features if returnList else np.ravel(hog_features)
    # return hog_features if returnList else np.array(hog_features)

def hog_vis(img,
    orientations=8,
    pxs_per_cell=8,
    cells_per_blk=2,
    feature_vector=False):
    ''' Histogram of Oriented Gradients, visualise=True 
    '''
    result = skFeat.hog(img,
        orientations=orientations,
        pixels_per_cell=(pxs_per_cell,pxs_per_cell),
        cells_per_block=(cells_per_blk,cells_per_blk),
        transform_sqrt=True, visualise=True, feature_vector=feature_vector
    )
    return SNS(
        features=result[0],
        images=result[1],
    )

def image_features(img, 
    spatial_size=None, # to add spatial feature, pass eg. (32,32)
    hist_bins=0,       # to add histogram feature, pass number of bins
    hist_ranges=None,  #  or pass ranges
    hog_params=None,   # to add hog feature, pass params dict for hog()
    ):
    features = []
    if spatial_size:
        features.append(spatial_features(img, spatial_size))
    if hist_bins or hist_ranges:
        features.append(color_hist(img, hist_bins, hist_ranges).hist)
    if hog_params:
        features.append(hog(img,
            hog_params.get('orientations', 8),
            hog_params.get('pxs_per_cell', 8),
            hog_params.get('cells_per_blk', 2),
            hog_params.get('channels', 'all'),
            hog_params.get('visualise', False),
            hog_params.get('feature_vector', False)))
    return np.concatenate(features)

def images_features(imgspath, 
    color_space='',    # color space to convert to, eg. 'YCrCb'
    spatial_size=None, # to add spatial feature, pass eg. (32,32)
    hist_bins=0,       # to add histogram feature, pass number of bins
    hist_ranges=None,  #  or pass ranges
    hog_params=None,   # to add hog feature, pass params dict for hog()
    ):
    ret = []
    for imgpath in imgspath:
        img = BGRto(color_space, cv2.imread(imgpath))
        ret.append(image_features(img, spatial_size, hist_bins, hist_ranges, hog_params))
    return ret

def add_heat(img=None, img_shape=None, mod_img=False, bboxes=[]):
    ''' Add heat to img 
    '''
    if mod_img:
        out = img
    else:
        out = np.zeros(img_shape) if img_shape else np.zeros_like(img[:,:,0])
    for box in bboxes:
        out[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return out

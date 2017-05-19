from glob import glob
from os.path import join
from types import SimpleNamespace as SNS

pklpath = SNS(
    svc='svc-all3-hog12-luv.pkl',
    scaler='scaler-all3-hog12-luv.pkl',
    # svc='pkls/svc-all3-hog12.pkl',
    # scaler='pkls/Xscaler-all3-hog12.pkl',
    # svc='svc-nohist.pkl',
    # X_scaler='Xscaler-nohist.pkl',
    # svc='svc-hogonly.pkl',
    # X_scaler='Xscaler-hogonly.pkl',
)
imgspath = '../data'
cars_imgspath = glob(join(imgspath, 'vehicles', '*/*.png'))
notcars_imgspath = glob(join(imgspath, 'non-vehicles', '*/*.png'))

default = SNS(
    # color_space='YCrCb',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    color_space='LUV',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 12,  # HOG orientations
    pix_per_cell = 8,  # HOG pixels per cell
    cell_per_block = 2,  # HOG cells per block
    hog_channel = 'ALL',  # Can be 0, 1, 2, or "ALL"
    train_size = (64, 64),  # train image size
    spatial_size = (32, 32),  # Spatial binning dimensions
    hist_bins = 32,  # Number of histogram bins
    spatial_feat = True,
    hist_feat = True,
    hog_feat = True,
)
defaults = {
    'color_space':default.color_space,
    'orient':default.orient,
    'pix_per_cell':default.pix_per_cell,
    'cell_per_block':default.cell_per_block,
    'hog_channel':default.hog_channel,
    'spatial_size':default.spatial_size,
    'hist_bins':default.hist_bins,
    'spatial_feat':default.spatial_feat,
    'hist_feat':default.hist_feat,
    'hog_feat':default.hog_feat,
}

import cv2
from lib import feature_extraction as fe
from config import imgspath


pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'
img = cv2.imread(imgspath+'/vehicles/GTI_Right/image0693.png')

for orient in range(9,13):
    hog = fe.hog_vis(img, orient, pix_per_cell, cell_per_block, hog_channel)
    out = cv2.resize(hog, (256,256), interpolation=cv2.INTER_AREA)
    cv2.imwrite('output_images/hog%d.jpg'%orient, out)

import cv2
import numpy as np
from lib import feature_extraction as fe
from lib import np_util as npu 
from lib import draw

image = cv2.imread('test_images/test1.jpg')
shape = image.shape
max_h = 330
win_rows = fe.bbox_rows(shape, max_h=max_h)

print('%d rows of windows' % len(win_rows))
img = draw.boxes_list(image, win_rows)
cv2.imwrite('output_images/sliding_windows_%d.jpg'%max_h, img)

for i,wins in enumerate(win_rows):
    img = draw.boxes(image, wins)
    cv2.imwrite('output_images/sliding_windows_%d-%d.jpg'%(max_h,i), img)
    print('window size: ', wins[0][1][0] - wins[0][0][0])

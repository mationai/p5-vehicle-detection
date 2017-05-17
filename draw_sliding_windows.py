import cv2
import numpy as np
from lib.detection import slide_window_rows
from toolbox import draw
from lib import np_util as npu 

image = cv2.imread('test_images/test1.jpg')
shape = image.shape
win_rows = slide_window_rows(shape)

print('%d rows of windows' % len(win_rows))
img = draw.boxes_list(image, win_rows)
cv2.imwrite('output_images/sliding_windows.jpg', img)

for i,wins in enumerate(win_rows):
    img = draw.boxes(image, wins)
    cv2.imwrite('output_images/sliding_windows%d.jpg'%i, img)

import cv2
import numpy as np
from lib.sliding_window import *
from toolbox.draw_on_image import draw_boxes

colors = [
 (0,255,0),
 (0,215,0),
 (0,175,0),
 (0,135,0),
 (0,100,0),
 (0,70,0),
 (0,40,0),
 (0,10,0),
]
image = cv2.imread('test_images/test1.jpg')
shape = image.shape

for i,shift in enumerate(slide_windows(shape)):
    draw_image = np.copy(image)
    # print(len(shift),'\n',i)
    for j,stripe in enumerate(shift):
        draw_image = draw_boxes(draw_image, stripe, colors[j])
    # cv2.imshow('shifted_collection', draw_image)
    cv2.imwrite('output_images/sliding_windows%d.jpg'%i, draw_image)

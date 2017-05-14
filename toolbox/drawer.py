import cv2
import numpy as np
from toolbox import color_palette as cp


def draw_boxes(img, bboxes=[], colors=cp.greens, thick=2, return_copy=True):
    ''' Returns img with boxes drawn 
    bboxes: list of bounding box coords
    thick: line thickness
    '''
    out = np.copy(img) if return_copy else img
    colorslen = len(colors)
    for i,box in enumerate(bboxes):
        cv2.rectangle(out, box[0], box[1], colors[i%colorslen], thick)
    return out

def draw_boxes_list(img, bboxeslist=[], palette=cp.palette, thick=2, return_copy=True):
    out = np.copy(img) if return_copy else img
    colorslen = len(palette)
    for i,bboxes in enumerate(bboxeslist):
        j = i%colorslen
        draw_boxes(out, bboxes, palette[i%colorslen], thick=thick, return_copy=False) 
    return out

def draw_labeled_boxes(img, bboxes=None, colors=[(0,0,255)], thick=2, labels=None):
    ''' Returns img with boxes drawn 
    bboxes: list of bound boxes
    color: color of box line
    thick: line thickness
    labels: (labelsAry, count) tuple from scipy.ndimage.measurements.label
    '''
    for n in range(1, labels[1]+1):
        nonzero = (labels[0]==n).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        j = (n-1)%colorslen
        cv2.rectangle(img, bbox[0], bbox[1], colors[j], thick)
    return img

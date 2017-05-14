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

def draw_sidewins(imgs, title_texts):
    ''' Take upto 3 wins, resize them smaller so they can be vertically stacked.
    Returns a long image that is longer than each win's original ht such that 
    its ht will equal to main image's ht plus another bottom debug window.
    ''' 
    xratio = 0.32
    yratio = 0.34
    vis = [cv2.resize(img, None,None,xratio,yratio,cv2.INTER_AREA) for img in imgs] 
    ht, wd = vis[0].shape[:2]
    title_ht = ht//5
    txtpos = (10, title_ht-15)
    size = 1 
    fontwd = 2
    linetype = cv2.LINE_AA 
    titles = [np.zeros((title_ht, wd, 3)).astype(np.uint8) for img in imgs]
    outimgs = []
    for i,title in enumerate(titles):
        title[:,:,0] = 255
        title[:,:,1] = 255
        title[:,:,2] = 255
        cv2.putText(title,title_texts[i],txtpos,font,size,(0,0,0),fontwd,linetype)
        outimgs.append(title)
        outimgs.append(vis[i])
    return np.vstack(outimgs)

def add_btm_textwin(img, texts):
# def addBtmText(img, offcent, radm, fitmsg, detectmsg):
#     ''' Returns stacked main image and bottom texts
#     ''' 
#     txt_row_ht = 25

#     txtimg = np.zeros((out_ht-img_ht, img_wd, 3)).astype(np.uint8)
#     radtxt = 'Curve Radius'
#     postxt = 'Car off'
#     toptxt = radtxt+'. '+postxt
#     lfit = 'Lf Fit: %s' % fitmsg[L]
#     rfit = 'Rt Fit: %s' % fitmsg[R]
#     ldetect = 'Lf Detect: %s' % detectmsg
#     rdetect = 'Rt Detect: %s' % detectmsg
#     x,y = 10,20
#     size = .5
#     ht = int(size * 45)
#     wd = 1
#     cv2.putText(txtimg, toptxt,(x,y), font, size, (255,255,255), wd,linetype)
#     cv2.putText(txtimg, lfit,   (x,y+ht*1),font,size,getColor(fitmsg[L]),wd,linetype)
#     cv2.putText(txtimg, rfit,   (x,y+ht*2),font,size,getColor(fitmsg[R]),wd,linetype)
#     cv2.putText(txtimg, ldetect,(x,y+ht*3),font,size,(255,255,255),wd,linetype)
#     cv2.putText(txtimg, rdetect,(x,y+ht*4),font,size,(255,255,255),wd,linetype)
#     return np.vstack((img, txtimg))

    return np.vstack((img, img))

def add_debug_wins(img, dbg):
# def add_debug_wins(img, imgs, title_texts, btm_texts):
    sidewin = draw_sidewins(dbg.imgs, dbg.title_texts)
    mainwin = add_btm_textwin(img, dbg.btm_texts)
    return np.hstack((mainwin, sidewin))

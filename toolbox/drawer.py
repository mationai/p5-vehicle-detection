import cv2
import numpy as np
from toolbox import color_palette as cp


font = cv2.FONT_HERSHEY_SIMPLEX
fontsize = .5
fontwd = 1
linetype = cv2.LINE_AA 


def heatmap(bboxes, img=None, shape=None):
    ''' Draw heatmap (filled bounding boxes) on copy of img or new image of shape.
    '''
    if img!=None:
        out = np.copy(img)
    elif shape:
        out = np.zeros(shape)
    else:
        raise ValueError('img or shape is required')
    for box in bboxes:
        out[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return out

def heat_overlay(bboxes, img, overlay=None, color=(255,0,0), alpha=.07):
    ''' Draw transparent heatmap (filled bounding boxes) on img.
    '''
    out = np.copy(img)
    if overlay==None:
        overlay = np.copy(img)
    for box in bboxes:
        cv2.rectangle(overlay, box[0], box[1], (0,255,0), cv2.FILLED)
        out = cv2.addWeighted(overlay, alpha, out, 1-alpha, 0)
    return out

def textrow_ht_y0(fontsize=fontsize):
    ''' Returns correct ht for text image rows. Needed as VideoFileClip.fl_image
    fails to write corect video if result ht is odd
    '''
    ht = 18 + int(fontsize*20)
    y0 =  9 + int(fontsize*20)
    return (ht, y0)

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
    ''' Calls draw_boxes() on list of bounding boxes.
    '''
    out = np.copy(img) if return_copy else img
    colorslen = len(palette)
    for i,bboxes in enumerate(bboxeslist):
        j = i%colorslen
        draw_boxes(out, bboxes, palette[i%colorslen], thick=thick, return_copy=False) 
    return out

def with_btm_win(img, texts, colors=[(255,255,255)]):
    ''' Returns stacked image of img and texts stacked vertically
    ''' 
    img_h, img_w = img.shape[:2]
    if len(colors)==1 and len(texts) > 1:
        colors *= len(texts) 
    txt_h, y0 = textrow_ht_y0()
    x = 8
    btm = np.zeros((txt_h*len(texts), img_w, 3)).astype(np.uint8)
    for i,txt in enumerate(texts):
        cv2.putText(btm,txt,(x,y0+(txt_h-2)*i),font,fontsize,colors[i],fontwd,linetype)
    return np.vstack((img, btm))

def side_wins(wins, img_shape, win_shape, titles=[''], wins_cnt=3):
    ''' Returns a vertically stacked image of up to maximgs reduced versions of wins.
    wins: images to be resized and vertically stacked, can be empty [].
    titles: text titles for the wins. Can be a list of single title for all wins. 
    img_shape: shape of main image. Its ht will be the ht of stacked images.
    win_shape: shape of win images before size reduction.
    ''' 
    img_h, img_w = img_shape[:2]
    org_win_h, org_win_w = win_shape[:2]
    aspect = org_win_w/org_win_h 
    txt_h, y0 = textrow_ht_y0()
    txtpos = (8, y0)

    desired_winlabel_h = img_h/wins_cnt
    _label_h = txt_h + 5
    win_h = int(desired_winlabel_h - _label_h +.5)
    win_w = int(win_h*aspect +.5)

    pxs_delta = img_h - win_h*wins_cnt - _label_h*wins_cnt
    delta = -1 if pxs_delta < 0 else 1
    label_h = [_label_h]*wins_cnt
    for i in range(abs(pxs_delta)):
        label_h[i] += delta

    lb_imgs = [np.zeros((label_h[i], win_w, 3)).astype(np.uint8) for i in range(wins_cnt)]
    out = []
    for i,lb_img in enumerate(lb_imgs):
        lb_img[:,:,None] = 255
        out.append(lb_img)
        if wins[i]==None:
            out.append(np.zeros((win_h, win_w, 3)).astype(np.uint8))
        else:
            out.append(cv2.resize(wins[i],(win_w,win_h),interpolation=cv2.INTER_AREA))
        cv2.putText(lb_img,titles[i],txtpos,font,fontsize,(0,0,0),fontwd,linetype)
    return np.vstack(out)

def with_debug_wins(img, btm_texts, wins, win_titles, wins_cnt=3):
    main = with_btm_win(img, btm_texts)
    win_shape = wins[0].shape if wins else img.shape
    side = side_wins(wins, main.shape, win_shape, win_titles, wins_cnt)
    # print(main.shape, side.shape)
    return np.hstack((main, side))

import cv2
import numpy as np
from toolbox import color_palette as cp


font = cv2.FONT_HERSHEY_SIMPLEX
fontsize = .5
fontwd = 1
linetype = cv2.LINE_AA 

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

def add_btm_win(img, texts, colors=[(255,255,255)]):
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

def draw_sidewins(wins, titles, img_shape, win_shape, wins_cnt=3):
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

    _win_w_lbl_h = img_h/wins_cnt
    _lb_h = txt_h + 5
    _win_h = _win_w_lbl_h - _lb_h
    winwd = int(_win_h*aspect +.5)
    winht = int(_win_h +.5)
    ratio = winht/org_win_h

    pxs_delta = img_h - winht*wins_cnt - _lb_h*wins_cnt
    delta = -1 if pxs_delta < 0 else 1
    lb_h = [_lb_h]*wins_cnt
    for i in range(abs(pxs_delta)):
        lb_h[i] += delta

    wins = [cv2.resize(img, None,None,ratio,ratio,cv2.INTER_AREA) for img in wins] 
    lb_imgs = [np.zeros((lb_h[i], winwd, 3)).astype(np.uint8) for i in range(wins_cnt)]

    if len(titles)==1:
        titles = [titles[0]+' %d'%(i+1) for i in range(wins_cnt)] 

    out = []
    for i,lb_img in enumerate(lb_imgs):
        lb_img[:,:,None] = 255
        cv2.putText(lb_img,titles[i],txtpos,font,fontsize,(0,0,0),fontwd,linetype)
        out.append(lb_img)
        if i < len(wins):
            out.append(wins[i])
        else:
            out.append(np.zeros((winht, winwd, 3)).astype(np.uint8))
    return np.vstack(out)

def add_debug_wins(img, texts, sideimgs, sidetitles):
    mainwin = add_btm_win(img, texts)
    sidewin = draw_sidewins(sideimgs, sidetitles, mainwin.shape, img.shape)
    return np.hstack((mainwin, sidewin))

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
    ht = 18 + fontsize*20
    y0 =  9 + fontsize*20
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
    out = np.copy(img) if return_copy else img
    colorslen = len(palette)
    for i,bboxes in enumerate(bboxeslist):
        j = i%colorslen
        draw_boxes(out, bboxes, palette[i%colorslen], thick=thick, return_copy=False) 
    return out

def add_btm_win(img, texts, colors=[(255,255,255)]):
# def addBtmText(img, offcent, radm, fitmsg, detectmsg):
#     ''' Returns stacked main image and bottom texts
#     ''' 
    imght, imgwd = img.shape[:2]
    if len(colors)==1 and len(texts) > 1:
        colors *= len(texts) 
    text_ht, y0 = textrow_ht_y0()
    x = 8
    btm = np.zeros((text_ht*len(texts), imgwd, 3)).astype(np.uint8)
    for i,txt in enumerate(texts):
        cv2.putText(btm,txt,(x,y0+(text_ht-2)*i),font,fontsize,colors[i],fontwd,linetype)
    return np.vstack((img, btm))

def draw_sidewins(wins, titles, img_shape, win_shape, wins_cnt=3):
    ''' Returns a vertically stacked image of up to maximgs reduced versions of wins.
    wins: images to be vertically stacked, can be empty [].
    titles: text titles for the wins. Can be a list of single title for all wins. 
    img_shape: shape of main image. Its ht will be the ht of stacked images.
    win_shape: shape of win images before size reduction.
    ''' 
    imght, imgwd = img_shape[:2]
    winht_org, winwd_org = win_shape[:2]
    aspect = winwd_org/winht_org 
    text_ht, y0 = textrow_ht_y0()

    _win_w_title_ht = imght/wins_cnt
    _title_ht = text_ht + 5
    _winht = _win_w_title_ht - _title_ht
    winwd = int(_winht*aspect +.5)
    winht = int(_winht +.5)

    pxs_delta = imght - winht*wins_cnt - _title_ht*wins_cnt
    delta = -1 if pxs_delta < 0 else 1
    title_hts = [_title_ht]*wins_cnt
    for i in range(abs(pxs_delta)):
        title_hts[i] += delta

    ratio = winht/winht_org
    wins = [cv2.resize(img, None,None,ratio,ratio,cv2.INTER_AREA) for img in wins] 
    if len(titles)==1:
        titles = [titles[0]+' %d'%(i+1) for i in range(wins_cnt)] 
    titleimgs = [
        np.zeros((title_hts[i], winwd, 3)).astype(np.uint8) for i in range(wins_cnt)]
    txtpos = (8, y0)

    outimgs = []
    for i,titleimg in enumerate(titleimgs):
        titleimg[:,:,None] = 255
        # titleimg[:,:,0] = 255
        # titleimg[:,:,1] = 255
        # titleimg[:,:,2] = 255
        cv2.putText(titleimg,titles[i],txtpos,font,fontsize,(0,0,0),fontwd,linetype)
        outimgs.append(titleimg)
        if i < len(wins):
            outimgs.append(wins[i])
        else:
            outimgs.append(np.zeros((winht, winwd, 3)).astype(np.uint8))
    return np.vstack(outimgs)

def add_debug_wins(img, texts, sideimgs, sidetitles):
    mainwin = add_btm_win(img, texts)
    sidewin = draw_sidewins(sideimgs, sidetitles, mainwin.shape, img.shape)
    return np.hstack((mainwin, sidewin))

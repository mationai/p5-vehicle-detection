import cv2
import numpy as np
from toolbox import color_palette as cp


font = cv2.FONT_HERSHEY_SIMPLEX
fontsize = .5
fontwd = 1
linetype = cv2.LINE_AA 


def heatmap(bboxes, img=None, shape=None):
    ''' Draw bounding boxes as heatmap on copy of img or new image if shape given.
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

def heatmap_on_color(bboxes, img=None, shape=None):
    ''' Draw bounding boxes as heatmap on copy of img or new image if shape given.
    '''
    if img!=None:
        out = np.copy(img)
    elif img_shape:
        out = np.zeros(img_shape)
    # elif img!=None:
    #     out = np.zeros_like(img[:,:,0])
    else:
        raise ValueError('img or img_shape is required')
    for box in bboxes:
        out[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
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

def side_wins(wins, titles, img_shape, win_shape, wins_cnt=3):
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

    # for win in wins:
    #     print(win.shape)
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

def with_debug_wins(img, btm_texts, sidewins, sidetitles, wins_cnt=3):
    main = with_btm_win(img, btm_texts)
    sidewin_shape = sidewins[0].shape if sidewins else img.shape
    sidewins = side_wins(sidewins, sidetitles, main.shape, sidewin_shape, wins_cnt)
    # sidewins = draw_sidewins(sideimgs, sidetitles, mainwin.shape, img.shape)
    return np.hstack((main, sidewins))

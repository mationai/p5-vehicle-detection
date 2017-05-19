from types import SimpleNamespace as SNS
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lib import np_util as npu
from lib import feature_extraction as fe
from lib.helpers import _x0,_x1,_y0,_y1
from toolbox import draw


def find_hot_boxes(img, box_rows, model, color_space='RGB',
                spatial_size=(32, 32), hist_bins=32, orient=9,
                pix_per_cell=8, cell_per_block=2,
                hog_channel=0, spatial_feat=True,
                hist_feat=True, hog_feat=True, hog_feats=None):
    ''' The "search_windows" function in lesson solution.
    Returns list of hot windows - windows which prediction is possitive for 
    box in windows.
    model: object with .classifier and .scaler
    '''
    result = []
    img = npu.RGBto(color_space, img)
    ymin, ymax = fe.ymin_ymax(box_rows)
    wins_cnt=0

    if hog_feat:
        # hog_roi_feats = fe.hog_features(img[ymin:ymax,:,:], orient,
        #     pix_per_cell, cell_per_block, hog_channel, feature_vec=False)
        hog_img_feats = fe.hog_features(img, orient, pix_per_cell, cell_per_block, 
            hog_channel, feature_vec=False)

    for row in box_rows:
        for box in row:
            win = BBox(box)
            test_img = cv2.resize(img[win.y0:win.y1, win.x0:win.x1], model.train_size)
            if hog_feat:
                x0 = win.x0//pix_per_cell
                y0 = win.y0//pix_per_cell
                wd = test_img.shape[1]//pix_per_cell - (cell_per_block-1) 
                # print(x0, y0, wd, box)
                # all still predicts mostly 1s
                hog_feats = hog_img_feats[:, y0:y0+wd, x0:x0+wd].ravel()
                # print('wd:',wd, 'y:',y0,y0+wd, 'x:',x0,x0+wd)
                # hog_feats = hog_img_feats[:, y0:y0+wd, x0:x0+wd,None].ravel()
                # hog_feats = hog_img_feats[:, y0:y0+wd, x0:x0+wd,:,:,:].ravel()

            features = fe.image_features(test_img, color_space=None,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, 
                                         spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feats=hog_feats,
                                         concat=True)
            test_features = model.scaler.transform(features.reshape(1, -1))
            prediction = model.classifier.predict(test_features)
            if prediction == 1:
                result.append(box)
    return result 

class BBox():
    def __init__(self, bbox=None, top_left=None, btm_right=None):
        ''' Either bbox OR top_left (and/or) btm_right needs to be passed.
        All are of ((x0,y0),(x1,y1))
        '''
        if bbox:
            self.x0 = _x0(bbox)
            self.x1 = _x1(bbox)
            self.y0 = _y0(bbox)
            self.y1 = _y1(bbox)
        else:
            btm_right = btm_right or top_left
            self.x0 = _x0(top_left)
            self.x1 = _x1(btm_right)
            self.y0 = _y0(top_left)
            self.y1 = _y1(btm_right)
        self.wd = self.x1 - self.x0

class Car():
    def __init__(self, bbox):
        ''' bbox is ((x0,y0),(x1,y1)) '''
        self.x0 = _x0(bbox)
        self.x1 = _x1(bbox)
        self.wins = [bbox]
        self.boxwd = self.x1 - self.x0

def car_winscnt_gen(cars, txt=''):
    return ("cars%s's windows counts: "%txt)+'; '.join([str(len(car.wins)) for car in cars])

dbg = SNS(
    crop = {
        'top':350,
        'btm':30,
        'left':360,
        'right':0,
    },
    wins_cnt = 3,
)
_colors = [(255,0,0),(0,0,255),(0,255,0)]
def _color(i):
    return _colors[i%len(_colors)] 

class CarsDetector():
    def __init__(self, model):
        self.img_shape = None
        self.rows = None
        self.cars = []
        self.model = model
        self.frame = 0

    def find_hot_wins(self, img):
        ''' Returns bounding windows of heats in img
        '''
        self.frame += 1
        if self.rows==None:
            self.rows = fe.bbox_rows(img.shape, dbg=True)
        dbgvis = np.copy(img)
        hot_boxes = find_hot_boxes(img, self.rows, self.model, **self.model.defaults)
        print('frame',self.frame, len(hot_boxes))
        heatmap = draw.heatmap(hot_boxes, shape=img.shape)
        hot_wins = fe.bboxes_of_heat(heatmap, threshold=1, frame=self.frame, dbg='a')
        dbgvis = draw.heat_overlay(hot_boxes, dbgvis)
        for win in hot_wins:
            dbgvis = draw.rect(dbgvis, win, (0,255,0))

        self.dbg_wins = [npu.crop(dbgvis, **dbg.crop)]
        self.side_txts = ['heat=new heats. box=window of heats']
        self.btm_txts = ['Frame %d'%self.frame, '%d new heatmap windows' % len(hot_wins)]
        return hot_wins

    def remove_if_detected(self, hot_wins):
                # Any heat bbox from new_heats will be moved to car.wins if condition checks. 
        # Note this loop won't run on 1st frame as self.cars is empty. 
        # "if new_heats" below will add cars to it if found
        for car in self.cars:
            for i in range(len(hot_wins))[::-1]:
                box = BBox(hot_wins[i])
                # print(car.x0, box.x1, car.x1, box.x0, car.boxwd*1.5 , box.wd)
                if car.x0 <= box.x1 and car.x1 >= box.x0 and car.boxwd*1.5 > box.wd:
                    car.wins.append(hot_wins.pop(i))

    def detect_cars(self, img):
        self.cars = sorted(self.cars, key=lambda car: car.boxwd, reverse=True)
        # sorted biggest to smallest
        dbgvis = np.copy(img)

        new_wins = self.find_hot_wins(img)
        len_raw_new_wins = len(new_wins)
        self.remove_if_detected(new_wins)
        self.side_txts.append('heat=not prv-detected. box=added for next filtering')

        new_cars = []
        if new_wins: # For those not removed, group them into as little number 
            # of cars as possible
            for win in new_wins:
                box = BBox(win)
                found = True
                for car in new_cars: 
                # loop won't run on first iteration as new_cars is empty.
                    if car.x0 <= box.x1 and car.x1 >= box.x0:
                        if car.boxwd*2 > box.wd:
                            car.wins.append(win)
                            found = False
                        break
                if found:
                    # 1st win is leftmost
                    new_cars.append(Car(win))
                    dbgvis = draw.rect(dbgvis, win, (0,255,0))

            new_cars = [car for car in new_cars if len(car.wins) > 0]
            self.cars.extend(new_cars)
            dbgvis = draw.heat_overlay(new_wins, dbgvis, alpha=.2)
            self.dbg_wins.append(npu.crop(dbgvis, **dbg.crop))
        else:
            self.dbg_wins.append(npu.crop(img, **dbg.crop))
        if new_cars:
            win_cnts = 'with '+' windows; '.join(['%d'%len(car.wins) for car in new_cars])+' windows '
        else:
            win_cnts = ''
        self.btm_txts.append('%d removed as similar to detected' % (len_raw_new_wins - len(new_wins)))
        self.btm_txts.append('%d removed as its the only heat' % (len_raw_new_wins - len(new_cars)))
        self.btm_txts.append('%d cars %sadded' % (len(new_cars), win_cnts))

    def detected_image(self, img):
        out = np.copy(img)
        dbgvis = np.copy(img)
        to_remove = []
        self.btm_txts.append(car_winscnt_gen(self.cars, '-for-detection'))

        # for i,car in enumerate(self.cars):
        #     if car.wins:
        #         car.wins.pop(0)   # always pop 1st (prv frame's) box for all cars
        #     if len(car.wins) < 1: # if 1 car heats after pop, remove it (tried < 3, worst)
        #         to_remove.append(i)
        # self.btm_txts.append('%s cars removed due to no windows found' % (len(to_remove) or 'No'))

        # for i in to_remove[::-1]:
        #     self.cars.pop(i)

        for i,car in enumerate(self.cars):
            # while len(car.wins) > 30:
            while len(car.wins) > 24:
            # while len(car.wins) > 40:
                car.wins.pop(0)
            # if len(car.wins) > 14:
            if len(car.wins) > 12:
                heatmap = draw.heatmap(car.wins, shape=img.shape)
                hot_wins = fe.bboxes_of_heat(heatmap, threshold=5, frame=self.frame)
                if hot_wins:
                    box = hot_wins[0]
                    if car.boxwd > 20:
                        out = draw.rect(out, box, (0,255,0))
                        dbgvis = draw.rect(dbgvis, box, _color(i))
                dbgvis = draw.heat_overlay(car.wins, dbgvis, alpha=.04)

            if car.wins:
                car.wins.pop(0)   # always pop 1st (prv frame's) box for all cars
            if len(car.wins) < 1: # if 1 car heats after pop, remove it (tried < 3, worst)
                to_remove.append(i)

        for i in to_remove[::-1]:
            self.cars.pop(i)

        self.btm_txts.append('%s cars removed due to no windows found' % (len(to_remove) or 'No'))
        self.dbg_wins.append(npu.crop(dbgvis, **dbg.crop))
        self.side_txts.append('detected cars')
        # print(len(self.btm_txts))
        return draw.with_debug_wins(out, self.btm_txts, self.dbg_wins, self.side_txts, dbg.wins_cnt)

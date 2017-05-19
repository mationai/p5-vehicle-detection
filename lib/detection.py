from types import SimpleNamespace as SNS
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lib import np_util as npu
from lib import feature_extraction as fe
from lib.helpers import _x0,_x1,_y0,_y1
from toolbox import draw


def hot_rows(img, box_rows, model, color_space='RGB',
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
        hot = []

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
                hot.append(box)
            wins_cnt += 1
        result.append(hot)
    # print(len(box_rows), wins_cnt)
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
        self.heats = [bbox]
        self.boxwd = self.x1 - self.x0

def car_winscnt_gen(cars, txt=''):
    return ("cars%s's windows count "%txt)+', '.join([str(len(car.heats)) for car in cars])

dbg = SNS(
    crop = {
        'top':350,
        'btm':10,
        'left':330,
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

    def find_hot_boxes(self, img):
        if self.rows==None:
            self.rows = fe.sliding_box_rows(img.shape, dbg=True)
        dbgvis = np.copy(img)
        hot_boxes = []
        for hot_row in hot_rows(img, self.rows, self.model, **self.model.defaults):
            heatmap = draw.heatmap(hot_row, shape=img.shape)
            hot_boxes += fe.bboxes_of_heat(heatmap, threshold=2)
            dbgvis = draw.heat_overlay(hot_row, dbgvis)
        for box in hot_boxes:
            dbgvis = draw.rect(dbgvis, box, (0,255,0))

        self.dbg_wins = [npu.crop(dbgvis, **dbg.crop)]
        self.side_txts = ['heat=new heats. box=bbox of heats']
        self.btm_txts = ['Frame %d'%self.frame, '%d new heatmap windows' % len(hot_boxes)]
        return hot_boxes

    def remove_if_detected(self, hot_boxes):
                # Any heat bbox from new_heats will be moved to car.heats if condition checks. 
        # Note this loop won't run on 1st frame as self.cars is empty. 
        # "if new_heats" below will add cars to it if found
        for car in self.cars:
            for i in range(len(hot_boxes))[::-1]:
                box = BBox(hot_boxes[i])
                # print('car.x0', car.x0, '' <= box.x1 and car.x1 >= box.x0 and car.boxwd*1.5 > box.wd:
                if car.x0 <= box.x1 and car.x1 >= box.x0 and car.boxwd*1.5 > box.wd:
                    car.heats.append(hot_boxes.pop(i))

    def detect_cars(self, img):
        self.frame += 1
        self.cars = sorted(self.cars, key=lambda car: car.boxwd, reverse=True)
        dbgvis = np.copy(img)

        hot_boxes = self.find_hot_boxes(img)
        new_detected = len(hot_boxes)
        self.remove_if_detected(hot_boxes)
        self.side_txts.append('heat=not prv-detected. box=added for next filtering')

        new_cars = []
        if hot_boxes: # For those not removed, group them into as little number 
            # of cars as possible
            for box in hot_boxes:
                heat = BBox(box)
                found = True
                for car in new_cars: 
                # loop won't run on first heat as new_cars is empty.
                    if car.x0 <= heat.x1 and car.x1 >= heat.x0:
                        if car.boxwd*2 > heat.wd:
                            car.heats.append(box)
                            found = False
                        break
                if found:
                    new_cars.append(Car(box))
                    dbgvis = draw.rect(dbgvis, box, (0,255,0))

            new_cars = [car for car in new_cars if len(car.heats) > 1]
            self.cars.extend(new_cars)
            dbgvis = draw.heat_overlay(hot_boxes, dbgvis, alpha=.2)
            self.dbg_wins.append(npu.crop(dbgvis, **dbg.crop))
        else:
            self.dbg_wins.append(npu.crop(img, **dbg.crop))
        if new_cars:
            win_cnts = 'with '+', '.join(['%d'%len(car.heats) for car in new_cars])+' windows '
        else:
            win_cnts = ''
        self.btm_txts.append('%d removed (previously detected)' % (new_detected - len(hot_boxes)))
        self.btm_txts.append('%d cars %sadded' % (len(new_cars), win_cnts))

    def detected_image(self, img):
        out = np.copy(img)
        dbgvis = np.copy(img)
        to_remove = []
        self.btm_txts.append(car_winscnt_gen(self.cars, '-for-detection'))

        for i,car in enumerate(self.cars):
            if car.heats:
                car.heats.pop(0)   # always pop 1st (prv frame's) box for all cars
            if len(car.heats) < 1: # if 1 car heats after pop, remove it (tried < 3, worst)
                to_remove.append(i)
        self.btm_txts.append('%s cars removed due to no windows found' % (len(to_remove) or 'No'))

        for i in to_remove[::-1]:
            self.cars.pop(i)

        for i,car in enumerate(self.cars):
            # while len(car.heats) > 30:
            while len(car.heats) > 24:
            # while len(car.heats) > 40:
                car.heats.pop(0)
            # if len(car.heats) > 14:
            if len(car.heats) > 12:
                heatmap = draw.heatmap(car.heats, shape=img.shape)
                hot_boxes = fe.bboxes_of_heat(heatmap, threshold=5)
                if hot_boxes:
                    box = hot_boxes[0]
                    if car.boxwd > 20:
                        out = draw.rect(out, box, (0,255,0))
                        dbgvis = draw.rect(dbgvis, box, _color(i))
                dbgvis = draw.heat_overlay(car.heats, dbgvis, alpha=.04)

        self.dbg_wins.append(npu.crop(dbgvis, **dbg.crop))
        self.side_txts.append('detected cars')
        # print(len(self.btm_txts))
        return draw.with_debug_wins(out, self.btm_txts, self.dbg_wins, self.side_txts, dbg.wins_cnt)

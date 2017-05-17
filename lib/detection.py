from types import SimpleNamespace as SNS
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lib import np_util as npu
from lib import feature_extraction as fe
from lib.helpers import _x0,_x1,_y0,_y1
from toolbox import drawer as draw


def hot_win_rows(img, window_rows, model, color_space='RGB', min_w=64,
                spatial_size=(32, 32), hist_bins=32,
                hist_range=(0, 256), orient=9,
                pix_per_cell=8, cell_per_block=2,
                hog_channel=0, spatial_feat=True,
                hist_feat=True, hog_feat=True):
    ''' The "search_windows" function in lesson solution.
    Returns list of hot windows - windows which prediction is possitive for 
    window in windows.
    model: object with .classifier and .X_scaler
    '''
    outrows = []
    img = npu.RGBto(color_space, img)

    # 2) Iterate over all windows in the list
    for row in window_rows:
        hot_wins = []
        box = Box(top_left=row[0], btm_right=row[-1])
        roi = img[box.y0:box.y1, box.x0:box.x1]
        ht, wd = roi.shape[:2]
        scale = min(ht, wd)/min_w
        roi = cv2.resize(roi, (int(wd/scale), int(ht/scale)))

        if hog_feat:
            hog_features = fe.hog_features(roi, orient, pix_per_cell, 
                cell_per_block, hog_channel, feature_vec=False)
        for window in row:
            # 3) Extract the test window from original image
            x0 = int(_x0(window)/scale)
            # resized_win_start = int(window[0][0]/scale)
            if wd < x0 + min_w:
                x0 = wd - min_w

            test_img = np.array(roi)[:,x0:x0+min_w]
            # 4) Extract features for that window using single_img_features()
            features = fe.image_features(test_img, color_space=None,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=False)
            if hog_feat:
                #print(window,scale,resized_win_start)
                hog_x0 = int(x0/pix_per_cell)
                nblocks = int(min_w/pix_per_cell) - (cell_per_block-1)
                hog_feature = np.array(hog_features)[:,hog_x0:hog_x0+nblocks].ravel()
                features.append(hog_feature)

            features = np.concatenate(features)
            # 5) Scale extracted features to be fed to classifier
            test_features = model.X_scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = model.classifier.predict(test_features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                hot_wins.append(window)
        outrows.append(hot_wins)
    # 8) Return windows for positive detections
    return outrows

class Box():
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

def btm_text_gen(cars, txt=''):
    return 'car windows count '+txt+' '.join([str(len(car.wins)) for car in cars])

dbg = SNS(
    crop={
        'top':350,
        'btm':10,
        'left':300,
        'right':340,
        # 'left':200,
        # 'right':100,
    },
    wins_cnt=3,
)
class CarsDetector():
    def __init__(self, model):
        self.img_shape = None
        self.rows = None
        self.cars = []
        self.model = model

    def find_heat_boxes(self, img):
        if self.rows==None:
            self.rows = fe.sliding_box_rows(img.shape, dbg=True)
        dbg_heat = np.copy(img)
        dbg_overlay = np.copy(img)
        hot_wins = []
        for hot_row in hot_win_rows(img, self.rows, self.model, **self.model.defaults):
            heatmap = draw.heatmap(hot_row, shape=img.shape)
            hot_wins += fe.get_bboxes(heatmap, threshold=2)
            dbg_heat = draw.heat_overlay(hot_row, dbg_heat, dbg_overlay)

        self.dbg_wins = [npu.crop(dbg_heat, **dbg.crop)]
        self.dbg_lbls = ['new unfiltered heats']
        return hot_wins

    def detect_cars(self, img):
        new_heats = self.find_heat_boxes(img)
        self.cars = sorted(self.cars, key=lambda car: car.boxwd, reverse=True)
        self.btm_txts = ['detection coords: ']

        # Move heats from new_heats to car.wins if condition checks. 
        # this loop won't run on 1st frame as self.cars is empty. 
        # "if new_heats" below will add cars to it if found
        for car in self.cars:
            for i in range(len(new_heats))[::-1]:
                box = Box(new_heats[i])
                # print('car.x0', car.x0, '' <= box.x1 and car.x1 >= box.x0 and car.boxwd*1.5 > box.wd:
                if car.x0 <= box.x1 and car.x1 >= box.x0 and car.boxwd*1.5 > box.wd:
                    car.wins.append(
                        new_heats.pop(i))
        if new_heats:
            dbg_heat = np.copy(img)
            cars = []
            for bbox in new_heats:
                heat = Box(bbox)
                found = True
                # cars is [] until it is filled in "if found" below
                for car in cars:
                    if car.x0 <= heat.x1 and car.x1 >= heat.x0:
                        if car.boxwd*2 > heat.wd:
                            car.wins.append(bbox)
                            found = False
                        break
                if found:
                    cars.append(Car(bbox))
                dbg_heat = cv2.rectangle(dbg_heat, bbox[0], bbox[1], (0,255,0), 2)
            self.cars.extend(cars)
            self.dbg_wins.append(npu.crop(dbg_heat, **dbg.crop))
            self.dbg_lbls.append('detections')
            self.btm_txts[0] += '; '.join(['(%d,%d),(%d,%d)'%(_x0(b),_y0(b),_x1(b),_y1(b)) for b in new_heats])

    def detected_image(self, img):
        out = np.copy(img)
        ghost_cars = []
        self.btm_txts.append(btm_text_gen(self.cars, 'detected: '))

        for i,car in enumerate(self.cars):
            if car.wins:
                car.wins.pop(0) # always pop 1st win for all car in self.cars
            if not car.wins:    # if car.wins count=1 (0 after pop), add i to ghost_cars
                ghost_cars.append(i)

        for ighost in ghost_cars[::-1]:
            self.cars.pop(ighost) # remove that 
        self.btm_txts.append(btm_text_gen(self.cars, 'filter1: '))

        for i,car in enumerate(self.cars):
            dbg_heat = np.copy(img)

            while len(car.wins) > 40:
                car.wins.pop(0)
            if len(car.wins) > 12:
                heatmap = draw.heatmap(car.wins, shape=img.shape)
                hot_wins = fe.get_bboxes(heatmap, threshold=5)
                if hot_wins:
                    heatbox = hot_wins[0]
                if car.boxwd > 20:
                    out = cv2.rectangle(out, heatbox[0], heatbox[1], (0,255,0), 2)
                if len(self.dbg_wins) < dbg.wins_cnt:
                    dbg_heat = draw.heat_overlay(car.wins, dbg_heat)
                    self.dbg_wins.append(npu.crop(dbg_heat, **dbg.crop))

                    self.dbg_lbls.append('detected cars')
        self.btm_txts.append(btm_text_gen(self.cars, 'filter2: '))
        # print(len(self.btm_txts))
        return draw.with_debug_wins(out, self.btm_txts, self.dbg_wins, self.dbg_lbls, dbg.wins_cnt)

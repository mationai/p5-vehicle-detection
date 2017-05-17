from types import SimpleNamespace as SNS
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lib import np_util as npu
from lib import feature_extraction as fe
from lib.helpers import _x0,_x1,_y0,_y1
from toolbox import draw


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

def car_winscnt_gen(cars, txt=''):
    return ("cars%s's windows count "%txt)+', '.join([str(len(car.wins)) for car in cars])

dbg = SNS(
    crop={
        'top':350,
        'btm':10,
        'left':330,
        'right':0,
    },
    wins_cnt=3,
)
class CarsDetector():
    def __init__(self, model):
        self.img_shape = None
        self.rows = None
        self.cars = []
        self.model = model

    def find_hot_wins(self, img):
        if self.rows==None:
            self.rows = fe.sliding_box_rows(img.shape, dbg=True)
        dbg_heat = np.copy(img)
        hot_wins = []
        for hot_row in hot_win_rows(img, self.rows, self.model, **self.model.defaults):
            heatmap = draw.heatmap(hot_row, shape=img.shape)
            hot_wins += fe.bound_wins(heatmap, threshold=2)
            dbg_heat = draw.heat_overlay(hot_row, dbg_heat)
        for win in hot_wins:
            dbg_heat = cv2.rectangle(dbg_heat, win[0], win[1], (0,255,0), 2)

        self.dbg_wins = [npu.crop(dbg_heat, **dbg.crop)]
        self.side_txts = ['heat=new heats. box=win of heats']
        self.btm_txts = ['%d new heatmap windows' % len(hot_wins)]
        return hot_wins

    def remove_if_detected(self, hot_wins):
                # Any heat bbox from new_heats will be moved to car.wins if condition checks. 
        # Note this loop won't run on 1st frame as self.cars is empty. 
        # "if new_heats" below will add cars to it if found
        for car in self.cars:
            for i in range(len(hot_wins))[::-1]:
                box = Box(hot_wins[i])
                # print('car.x0', car.x0, '' <= box.x1 and car.x1 >= box.x0 and car.boxwd*1.5 > box.wd:
                if car.x0 <= box.x1 and car.x1 >= box.x0 and car.boxwd*1.5 > box.wd:
                    car.wins.append(hot_wins.pop(i))

    def detect_cars(self, img):
        self.cars = sorted(self.cars, key=lambda car: car.boxwd, reverse=True)
        dbg_heat = np.copy(img)

        hot_wins = self.find_hot_wins(img)
        new_detected = len(hot_wins)
        self.remove_if_detected(hot_wins)
        self.side_txts.append('heat=not prv-detected. box=added for next filtering')

        new_cars = []
        if hot_wins: # For those not removed, group them into as little number 
            # of cars as possible
            for bbox in hot_wins:
                heat = Box(bbox)
                found = True
                for car in new_cars: 
                # loop won't run on first heat as new_cars is empty.
                    if car.x0 <= heat.x1 and car.x1 >= heat.x0:
                        if car.boxwd*2 > heat.wd:
                            car.wins.append(bbox)
                            found = False
                        break
                if found:
                    new_cars.append(Car(bbox))
                    dbg_heat = cv2.rectangle(dbg_heat, bbox[0], bbox[1], (0,255,0), 2)

            new_cars = [car for car in new_cars if len(car.wins) > 1]
            self.cars.extend(new_cars)
            dbg_heat = draw.heat_overlay(hot_wins, dbg_heat, alpha=.2)
            self.dbg_wins.append(npu.crop(dbg_heat, **dbg.crop))
        else:
            self.dbg_wins.append(npu.crop(img, **dbg.crop))
        if new_cars:
            win_cnts = 'with '+', '.join(['%d'%len(car.wins) for car in new_cars])+' windows '
        else:
            win_cnts = ''
        self.btm_txts.append('%d removed (previously detected)' % (new_detected - len(hot_wins)))
        self.btm_txts.append('%d cars %sadded' % (len(new_cars), win_cnts))

    def detected_image(self, img):
        out = np.copy(img)
        dbg_heat = np.copy(img)
        to_remove = []
        self.btm_txts.append(car_winscnt_gen(self.cars, '-for-detection'))

        for i,car in enumerate(self.cars):
            if car.wins:
                car.wins.pop(0)   # always pop 1st (prv frame's) win for all cars
            if len(car.wins) < 2: # if only 1 car wins after pop, add to remove list
                to_remove.append(i)
        self.btm_txts.append('%d cars removed due to no windows found' % len(to_remove))

        for i in to_remove[::-1]:
            self.cars.pop(i)

        for i,car in enumerate(self.cars):
            while len(car.wins) > 40:
                car.wins.pop(0)
            if len(car.wins) > 12:
                heatmap = draw.heatmap(car.wins, shape=img.shape)
                hot_wins = fe.bound_wins(heatmap, threshold=5)
                if hot_wins:
                    heatbox = hot_wins[0]
                if car.boxwd > 20:
                    out = cv2.rectangle(out, heatbox[0], heatbox[1], (0,255,0), 2)
                dbg_heat = draw.heat_overlay(car.wins, dbg_heat)

        self.dbg_wins.append(npu.crop(dbg_heat, **dbg.crop))
        self.side_txts.append('detected cars')
        self.btm_txts.append(car_winscnt_gen(self.cars, '-detected'))
        # print(len(self.btm_txts))
        return draw.with_debug_wins(out, self.btm_txts, self.dbg_wins, self.side_txts, dbg.wins_cnt)

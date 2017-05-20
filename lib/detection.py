from types import SimpleNamespace as SNS
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2, string
from lib import np_util as npu
from lib import feature_extraction as fe
from lib.helpers import _x0,_x1,_y0,_y1,_wd
from toolbox import draw


def find_hot_boxes(img, box_rows, model, color_space='RGB',
                spatial_size=(32, 32), hist_bins=32, orient=9,
                pix_per_cell=8, cell_per_block=2,
                hog_channel=0, spatial_feat=True,
                hist_feat=True, hog_feat=True, hog_feats=None):
    ''' Returns list of hot windows - windows which prediction is possitive for
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
            b = X0Y0(box)
            test_img = cv2.resize(img[b.y0:b.y1, b.x0:b.x1], model.train_size)
            if hog_feat:
                x0 = b.x0//pix_per_cell
                y0 = b.y0//pix_per_cell
                wd = test_img.shape[1]//pix_per_cell - (cell_per_block-1) 
                hog_feats = hog_img_feats[:, y0:y0+wd, x0:x0+wd].ravel()

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

class X0Y0():
    def __init__(self, bbox):
        ''' bbox of ((x0,y0),(x1,y1))
        '''
        self.x0 = _x0(bbox)
        self.x1 = _x1(bbox)
        self.y0 = _y0(bbox)
        self.y1 = _y1(bbox)
        self.wd = self.x1 - self.x0

def cars_wins_str(cars):
    return ' '.join([car.winsinfo_str for car in cars])

def cars_coords_str(cars):
    return ' '.join([car.coords_str for car in cars])

def coords_gen(wins, txt=''):
    return ('; '.join(['x0=%d, wd=%d'%(_x0(win), _wd(win)) for win in wins]))

dbg = SNS(
    crop = { # these work with 4,6,7 lines of btm text, but not 5
        'top':350,
        'btm':30,
        'left':60,
        'right':0,
    },
    wins_cnt = 3,
)
_colors = [(255,0,0),(0,0,255),(0,255,0)]
def _color(i):
    return _colors[i%len(_colors)] 

car_labels = string.ascii_uppercase[:26]
class Car():
    def __init__(self, b, ilabel):
        ''' bbox is ((x0,y0),(x1,y1)) '''
        self.x0 = _x0(b)
        self.x1 = _x1(b)
        self.wins = [b]
        self.nwins = [1]
        self.label = car_labels[ilabel%len(car_labels)]

    @property
    def wd(self):
        return self.x1 - self.x0

    @property
    def nframes(self):
        return len(self.nwins)

    def add(self, b):
        self.wins.append(b)
        self.nwins[-1] += 1
        if _x0(b) < self.x0:
            self.x0 = _x0(b)
        if _x1(b) > self.x1:
            self.x1 = _x1(b)

    def overlaps(self, b, div_by=3):
        l = self.wd//div_by
        x0 = self.x0
        x1 = self.x1
        return x0-l <= _x0(b) <= x0+l and x1-l <= _x1(b) <= x1+l

    def empty_frames(self, cnt):
        for i in range(-cnt, 0):
            if self.nwins[i]!=0:
                return False
        return True

    @property
    def winscnt_str(self):
        return self.label+':'+','.join([str(cnt) for cnt in self.nwins])

    @property
    def winsinfo_str(self):
        s = ''
        j = 0
        for cnt in self.nwins:
            if cnt:
                wds = [str(_wd(self.wins[i])) for i in range(j, j+cnt)]
                s += ('%dx'%cnt)+','.join(wds)+', '
                j += cnt
            else:
                s += '0,'
        return self.label+':'+s

    @property
    def coords_str(self):
        return self.label+': x0=%d, wd=%d'%(self.x0, self.wd)

class CarDetector():
    def __init__(self, model, img_wd):
        self.rows = None
        self.cars = []
        self.model = model
        self.frame = 0
        self.icar = -1
        self.iheat = -1
        self.max_carwd = img_wd//3

    def find_hot_wins(self, img):
        ''' Returns bounding windows of heats in img
        '''
        if self.rows==None:
            self.rows = fe.bbox_rows(img.shape, dbg=True)
        dbg_img = np.copy(img)
        hot_boxes = find_hot_boxes(img, self.rows, self.model, **self.model.defaults)
        heatmap = draw.heatmap(hot_boxes, shape=img.shape)
        hot_wins = fe.bboxes_of_heat(heatmap, threshold=1)
        dbg_img = draw.heat_overlay(hot_boxes, dbg_img)
        for win in hot_wins:
            dbg_img = draw.rect(dbg_img, win, (0,255,0))

        self.dbg_wins = [npu.crop(dbg_img, **dbg.crop)]
        self.side_txts = ['New heats. box=window of heats']
        self.btm_txts = ['Frame %d'%self.frame]
        self.btm_txts.append('%d new heatmap windows of %s'%(len(hot_wins), coords_gen(hot_wins))) 
        self.frame += 1
        return hot_wins

    def move_to_cars(self, hot_wins):
                # Any heat bbox from new_heats will be moved to car.wins if condition checks. 
        # Note this loop won't run on 1st frame as self.cars is empty. 
        # "if new_heats" below will add cars to it if found
        for car in self.cars:
            for i in range(len(hot_wins))[::-1]: # reverse needed, else "index out of range" after pop()
                # print(car.x0, box.x1, car.x1, box.x0, car.boxwd*1.5 , box.wd)
                # len(hot_wins)>i shouldn't be needed, but "index out of range"...
                #  in 7% of prob6.mp4 and frame 9 of prob9.mp4
                if car.overlaps(hot_wins[i]):
                # if len(hot_wins)>i and car.overlaps(hot_wins[i]):
                    car.add(hot_wins.pop(i))

    @property
    def next_icar(self):
        self.icar += 1
        return self.icar
        
    def detect(self, img):
        self.cars = sorted(self.cars, key=lambda car: car.x0)#, reverse=True)
        for car in self.cars:
            car.nwins.append(0)

        dbg_img = np.copy(img)
        new_wins = self.find_hot_wins(img)
        len_raw_new_wins = len(new_wins)
        self.move_to_cars(new_wins)
        self.side_txts.append('Added to cars. box=added')

        new_cars = []
        added_to = []
        too_wide = 0
        if new_wins: # For those not removed, group them into as little number 
            # of cars as possible
            for win in new_wins:
                if _wd(win) > self.max_carwd:
                    too_wide += 1
                    continue

                found = True
                for car in new_cars: 
                # loop won't run on first iteration as new_cars is empty.
                    if car.overlaps(win):
                        car.add(win)
                        if car.label not in added_to:
                            added_to.append(car.label)
                        found = False
                        break
                if found:
                    new_car = Car(win, self.next_icar)
                    new_cars.append(new_car)
                    dbg_img = draw.rect(dbg_img, win, (0,255,0), new_car.label)

            # new_cars = [car for car in new_cars if len(car.wins) > 0]
            self.cars.extend(new_cars)
            dbg_img = draw.heat_overlay(new_wins, dbg_img, alpha=.2)
            self.dbg_wins.append(npu.crop(dbg_img, **dbg.crop))
        else:
            self.dbg_wins.append(npu.crop(img, **dbg.crop))

        self.btm_txts.append('%d removed - wider than %d'%(too_wide, self.max_carwd))
        self.btm_txts.append('%d removed - added to win of cars: %s' % (
            (len_raw_new_wins - len(new_wins), ', '.join(added_to))))
        # self.btm_txts.append('%d removed as its the only heat' % (len_raw_new_wins - len(new_cars)))
        self.btm_txts.append('%d cars added: %s' % (len(new_cars), cars_coords_str(new_cars)))
        self.btm_txts.append("car wins cnt: "+cars_wins_str(self.cars))

    def purge(self):
        purged = []
        for i,car in enumerate(self.cars):
            # purge oldest frame
            if car.nframes > 9:
                while car.nwins[0] and car.wins:
                    car.wins.pop(0)
                    car.nwins[0] -= 1
                car.nwins.pop(0)

            # purge non-cars
            empty_cnt = 0
            if (3 <= car.nframes <= 5 and car.empty_frames(3)) or \
               (6 == car.nframes and car.empty_frames(4)) or \
               (7 <= car.nframes and car.empty_frames(5)):
                self.cars.pop(i)
                purged.append(car.label)
        self.btm_txts.append('%d cars purged: %s'%(len(purged), ', '.join(purged)))

    def detected_image(self, img):
        self.detect(img)
        self.purge()

        out = np.copy(img)
        dbg_img = np.copy(img)

        for i,car in enumerate(self.cars):
            if len(car.wins) > 1:
                heatmap = draw.heatmap(car.wins, shape=img.shape)
                wins_of_wins = None
                threshold = 7
                while not wins_of_wins and threshold > 5:
                    wins_of_wins = fe.bboxes_of_heat(heatmap, threshold)
                    threshold -= 1
                if len(wins_of_wins) > 1:
                    self.btm_txts[0] += ' MULTIPLE (non-overlapping?) wins for car-'+car.label
                for win in wins_of_wins:
                    out = draw.rect(out, win, (0,255,0), car.label)
                    dbg_img = draw.rect(dbg_img, win, _color(i), car.label)
                dbg_img = draw.heat_overlay(car.wins, dbg_img)

        self.dbg_wins.append(npu.crop(dbg_img, **dbg.crop))
        self.side_txts.append('detected cars')
        return draw.with_debug_wins(out, self.btm_txts, self.dbg_wins, self.side_txts, dbg.wins_cnt)

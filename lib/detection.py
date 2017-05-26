from types import SimpleNamespace as SNS
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2, string
from lib import np_util as npu
from lib import feature_extraction as fe
from lib import draw
from lib.helpers import _x0,_x1,_y0,_y1,_wd,_ht


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
    wins_cnt=0

    if hog_feat:
        # Tried to crop with img[360:] "ValueError: operands could not be broadcast together with shapes...",
        hog_img_feats = fe.hog_features(img, orient, pix_per_cell, cell_per_block, 
            hog_channel, feature_vec=False)

    for row in box_rows:
        for box in row:
            b = X0Y0(box)
            test_img = cv2.resize(img[b.y0:b.y1, b.x0:b.x1], model.train_size)
            if hog_feat:
                x0 = b.x0//pix_per_cell
                y0 = b.y0//pix_per_cell # - 360 if hog_roi_feats works
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

## Settings for side wins
dbg = SNS(
    crop = { # these settings work with 4,6,7 lines of btm text, fails on 5 lines
        'top':350,
        'btm':30,
        'left':0,
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
        ''' b: ((x0,y0),(x1,y1)) '''
        self.x0 = _x0(b)
        self.x1 = _x1(b)
        self.wins = [b]   # detected windows of heat
        self.nwins = [1]  # number of windows per frame
        self.label = car_labels[ilabel%len(car_labels)]
        self.lifetime = 1 # total number of frames

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

    def new_frame(self):
        self.nwins.append(0)
        self.lifetime += 1

    def pop_win(self):
        if self.x0 == _x0(self.wins[0]):
            self.x0 = np.min([_x0(self.wins[i]) for i in range(1, len(self.wins))])
        if self.x1 == _x1(self.wins[0]):
            self.x1 = np.max([_x1(self.wins[i]) for i in range(1, len(self.wins))])
        self.wins.pop(0)
        self.nwins[0] -= 1

    def overlaps(self, b):
        l = self.wd//3
        return self.x0-l <= _x0(b) <= self.x0+l

    def overlap_by(self, b):
        l = self.wd//3
        return self.x1-l <= _x1(b) <= self.x1+l

    def empty_frames(self, cnt):
        ''' Returns of number of consecutive no win-found frames is >= cnt
        ''' 
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
                s += '%d,'%_wd(self.wins[j]) if cnt==1 else '%d,'%cnt
                j += cnt
            else:
                s += '0,'
        return self.label+'('+s[:-1]+')'

    @property
    def coords_str(self):
        return self.label+': x0=%d, wd=%d'%(self.x0, self.wd)

class CarDetector():
    def __init__(self, model, img_shape):
        self.rows = None
        self.cars = []
        self.model = model
        self.frame = 0
        self.icar = -1
        self.iheat = -1
        self.max_carwd = img_shape[1]//3
        self.min_carwd = 64
        self.iffy_carht = img_shape[0]//3
        self.maxframes = 15

    def find_hot_wins(self, img):
        ''' Returns bounding windows of heats in img
            Updates self.dbg_wins and self.dbg_txts
        '''
        if self.rows==None:
            self.rows = fe.bbox_rows(img.shape)
        dbg_img = np.copy(img)
        hot_boxes = find_hot_boxes(img, self.rows, self.model, **self.model.defaults)
        heatmap = draw.heatmap(hot_boxes, shape=img.shape)
        hot_wins = fe.bboxes_of_heat(heatmap, threshold=2) #1 bad on prob10
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
        ''' Move wins in hot_wins to self.cars if conditions met
            Returns count of wins moved
        '''
        moved = 0
        for car in self.cars:
            for i in range(len(hot_wins))[::-1]: # "index out of range" after pop() if no reverse here
                if car.overlaps(hot_wins[i]) or car.overlap_by(hot_wins[i]):
                    car.add(hot_wins.pop(i))
                    moved += 1
        return moved

    @property
    def next_icar(self):
        ''' Increments self.icar and return it
        '''
        self.icar += 1
        return self.icar
        
    def detect(self, img):
        ''' Detect cars in img and add them to self.cars.
            Updates self.dbg_wins and self.dbg_txts
        '''
        for car in self.cars:
            car.new_frame()

        dbg_img = np.copy(img)
        new_wins = self.find_hot_wins(img)
        moved_cnt = self.move_to_cars(new_wins)
        self.side_txts.append('Cars to addd. heat=candidate box=added')

        new_cars = []
        added_to = []
        bad_size = []
        if new_wins: 
            # For those not removed, try grouping them into as least number of new_cars
            for win in new_wins:
                if (_wd(win) > self.max_carwd) or\
                   (_wd(win) < self.min_carwd) or\
                   (_ht(win) > self.iffy_carht and _ht(win) > _wd(win)):
                    bad_size.append(str(_wd(win)))
                    continue ## skip this win

                found = True
                for newcar in new_cars: 
                    # loop won't run on first iteration as new_cars is empty.
                    if newcar.overlaps(win): # overlap_by() not needed as new_cars are x0 ordered
                        if _wd(win) >= newcar.wd*1.5:
                            if newcar.label not in bad_size:
                                bad_size.append(newcar.label)
                        else:
                            newcar.add(win)
                            if newcar.label not in added_to:
                                added_to.append(newcar.label)
                        found = False
                        break
                if found:
                    new_car = Car(win, self.next_icar)
                    new_cars.append(new_car)
                    dbg_img = draw.rect(dbg_img, win, (0,255,0), new_car.label)

            self.cars.extend(new_cars)
            dbg_img = draw.heat_overlay(new_wins, dbg_img, alpha=.2)
            self.dbg_wins.append(npu.crop(dbg_img, **dbg.crop))
        else:
            self.dbg_wins.append(npu.crop(img, **dbg.crop))

        total = moved_cnt + len(bad_size)
        strs = []
        if bad_size:
            strs.append('%d bad size, widths are %s'%(len(bad_size), ', '.join(bad_size)))
        if moved_cnt:
            strs.append('%d added to wins of detected%s'%(moved_cnt, ', '.join(added_to)))
        self.btm_txts.append('%d removed: %s'%(total, ', '.join(strs)))
        self.btm_txts.append('%d cars added: %s' % (len(new_cars), cars_coords_str(new_cars)))
        self.btm_txts.append("car wins (width): "+cars_wins_str(self.cars))

    def purge(self):
        ''' Purge oldest frame from each detected car and remove cars with empty
             consecutive frames
            Returns list of text of purged info 
        '''
        purged = []
        for i,car in enumerate(self.cars):
            ## purge oldest frame
            if car.nframes > self.maxframes:
                while car.nwins[0] and car.wins:
                    car.pop_win()
                car.nwins.pop(0)

            ## remove cars with empty consecutive frames
            empty_cnt = 0
            if (3 <= car.nframes <= 5 and car.empty_frames(3)) or\
               (10 <= car.nframes and car.empty_frames(8)):
               # (12 <= car.nframes and car.empty_frames(10)): # tried, not as good
               # no need for (6==car.nframes and car.empty_frames(4)) .. 9 and (7) btw first and last check
                self.cars.pop(i)
                purged.append('%s(%d)'%(car.label,car.lifetime))
        return purged

    def final_purge_and_detection_image(self, img, purged):
        ''' Does some final purging and returns detection labeled image
            Updates self.dbg_wins and self.dbg_txts
        '''
        out = np.copy(img)
        dbg_img = np.copy(img)
        min_thres = 3

        for i,car in enumerate(self.cars):
            if len(car.wins) > min_thres:
                heatmap = draw.heatmap(car.wins, shape=img.shape)
                wins_of_wins = None
                threshold = 5

                while not wins_of_wins and threshold > min_thres:
                    wins_of_wins = fe.bboxes_of_heat(heatmap, threshold)
                    threshold -= 1

                ## continue means skip labeling as detected car 
                if not wins_of_wins:
                    continue

                ## wins shoudn't be disjoint, purge
                if len(wins_of_wins) > 1:
                    self.cars.pop(i)
                    purged.append('%s(disjoint wins)'%car.label)
                    continue

                win = wins_of_wins[0]
                wd, ht = _wd(win), _ht(win)
                frames_is_max = car.nframes >= self.maxframes

                ## Too small, purge 
                if wd < self.min_carwd and not frames_is_max:
                    self.cars.pop(i)
                    purged.append('%s(sml,w=%d)'%(car.label,wd))
                    continue

                ## Too narrow, purge
                if ht > wd and not frames_is_max:
                    self.cars.pop(i)
                    purged.append('%s(narrow,h=%d > w=%d)'%(car.label,ht,wd))
                    continue

                ## Too big, purge
                if ht >= self.iffy_carht and not frames_is_max:
                    self.cars.pop(i)
                    purged.append('%s(big,h=%d)'%(car.label,ht))
                    continue

                ## Alter win ht if too tall
                if ht >= wd*1.5 and frames_is_max:
                    win = (win[0], (_x1(win), _y0(win)+wd))

                out = draw.rect(out, win, (0,255,0), car.label)
                dbg_img = draw.rect(dbg_img, win, _color(i), car.label)
                dbg_img = draw.heat_overlay(car.wins, dbg_img)

        self.btm_txts.append('%d cars purged (lifetime/reason): %s'%(len(purged), ', '.join(purged)))
        self.dbg_wins.append(npu.crop(dbg_img, **dbg.crop))
        self.side_txts.append('detected cars')
        return draw.with_debug_wins(out, self.btm_txts, self.dbg_wins, self.side_txts, dbg.wins_cnt, {4:3})

    def detected_image(self, img):
        ''' API for running detect(), purge(), and final_purge_and_detection_image()
        '''
        self.detect(img)
        return self.final_purge_and_detection_image(img, self.purge())


from types import SimpleNamespace as SNS
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lib import np_util as npu
from lib import feature_extraction as fe
from lib.helpers import *
from toolbox.draw_on_image import *

x_,y_ = 0, 1

def horizontal_windows(img_shape, xmin=0, xmax=0, ymin=0, ymax=0,
                 winwd=64, winht=64, overlap=(0.5, 0.5)):
    ''' Returns list of window coords to be slide across image horizontally
    '''
    xmax = xmax or img_shape[1]
    ymax = ymax or img_shape[0]
    pxs_per_xstep = int(winwd * (1-overlap[x_]))
    pxs_per_ystep = int(winht * (1-overlap[y_]))

    # number of windows in x/y
    nx_windows = (xmax-xmin-winwd)//pxs_per_xstep + 1
    ny_windows = (ymax-ymin-winht)//pxs_per_ystep + 1

    windows = []
    for iy in range(ny_windows):
        for ix in range(nx_windows):
            x0 = ix * pxs_per_xstep + xmin
            x1 = x0 + winwd
            if x1 > img_shape[1]:
                x1 = x1 - (x1 - img_shape[1])
                x0 = x0 - (x1 - img_shape[1])
            y0 = iy * pxs_per_ystep + ymin
            y1 = y0 + winht
            windows.append(((x0, y0), (x1, y1)))
    return windows

def next_width(y, ymin=420, ymax=720, top_btm_wd=(32,320)):
    yposition_ratio = (y-ymin)/(ymax-ymin)
    width_diff = top_btm_wd[1] - top_btm_wd[0] 
    return int(yposition_ratio*width_diff + top_btm_wd[0])

def slide_windows(img_shape, ymin=440, ymax=None, maxht=.5, overlap=(0, 0.8), 
    shifts=1, minwd=64, dbg=False):
    ''' Returns strips of bounding box coords if shifts = 1.
    If shifts > 1, returns list of strips of bboxes.
    ymin: windows y start
    ymax: None = image ht
    maxht: max window ht in % of imght if <= 1, in pxs otherwise
    minwd: min window wd in pxs
    '''
    imght, imgwd = img_shape[:2]
    max_winwd = int(maxht*imght) if 0<=maxht<=1 else int(maxht) 
    # ymin = int(ymin_pct*imght) recog only 1 car in test1 for pct=.61 or .62
    ymax = ymax or imght
    strips_shifts = []
    _min_topwd = 32

    _xses = []
    for shift in range(shifts):
        y = ymax
        winwd = max_winwd
        windows_strips = []

        _xs = [shift]
        while (winwd >= minwd):
            # test1 print of overlap=(.9,.8), ymin=440,maxht=.5 (shifts d/c):
            # wd: 360, 276, 212, 163, 125,  97,  75
            #  y: 720, 649, 594, 552, 520, 496, 477
            x_step = int(winwd * (1-overlap[x_]))
            x = int(x_step * shift/shifts)
            y_step = int(winwd * (1-overlap[y_]))

            _xs.append((winwd, x_step, y_step, x))

            strip = horizontal_windows(img_shape, 
                xmin=x, xmax=imgwd, ymin=y-winwd, ymax=y, 
                winwd=winwd, winht=winwd, overlap=overlap)
            y -= y_step
            winwd = next_width(y, ymin, ymax, (_min_topwd, max_winwd))

            windows_strips.append(strip)
        strips_shifts.append(windows_strips)
        _xses.append(_xs)

    if dbg:
        print('shift, (winwd, x_step, y_step, x):')
        for xs in _xses:
            print(xs)
        by_wds = np.array(strips_shifts).T
        for by_wd in by_wds:
            win0 = by_wd[0][0]
            print('\nwidth %d:' % win0[1][0])
            for s in by_wd:
                for win in s:
                    print(win)
        colors = [
         (0,255,0),
         (0,215,0),
         (0,175,0),
         (0,135,0),
         (0,100,0),
         (0,70,0),
         (0,40,0),
         (0,10,0),
        ]
        for i,shift in enumerate(strips_shifts):
            draw_image = np.zeros(img_shape)
            # print(len(shift),'\n',i)
            for j,stripe in enumerate(shift):
                draw_image = draw_boxes(draw_image, stripe, colors[j])
            # cv2.imshow('shifted_collection', draw_image)
            cv2.imwrite('output_images/slide_windows%d.jpg'%i, draw_image)

    return strips_shifts if shifts>1 else strips_shifts[0]

def hot_windows(img, windows, clf, scaler, color_space='RGB',
                spatial_size=(32, 32), hist_bins=32,
                hist_range=(0, 256), orient=9,
                pix_per_cell=8, cell_per_block=2,
                hog_channel=0, spatial_feat=True,
                hist_feat=True, hog_feat=True):
    ''' aka "search_windows" in lesson solution.
    Returns list of hot windows - windows which prediction is possitive for 
    window in windows.
    '''
    result = []
    img = npu.RGBto(color_space, img)

    # 2) Iterate over all windows in the list
    for stripe in windows:
        stripe_on_windows = []
        current_stripe_area = ((stripe[0][0][0], stripe[0][0][1]), (stripe[-1][1][0], stripe[-1][1][1]))
        test_stripe = img[current_stripe_area[0][1]:current_stripe_area[1][1], current_stripe_area[0][0]:current_stripe_area[1][0]]
        scale = min(test_stripe.shape[0], test_stripe.shape[1]) / 64  # at most 64 rows and columns
        resized_test_stripe = cv2.resize(test_stripe,(np.int(test_stripe.shape[1] / scale), np.int(test_stripe.shape[0] / scale)))

        if hog_feat:
            hog_features = fe.hog_features(resized_test_stripe, orient, 
                pix_per_cell, cell_per_block, hog_channel, feature_vec=False)
        for window in stripe:
            # 3) Extract the test window from original image
            resized_win_start = int(window[0][0]/scale)
            if (resized_win_start + 64) > (resized_test_stripe.shape[1]):
                resized_win_start = resized_test_stripe.shape[1] - 64

            test_img = np.array(resized_test_stripe)[:, resized_win_start:(resized_win_start + 64)]
            #test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            features = fe.image_features(test_img, color_space=None,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=False)
            if hog_feat:
                #print(window,scale,resized_win_start)
                hog_win_start = int(resized_win_start / pix_per_cell)
                nblocks = int(64/pix_per_cell) - (cell_per_block-1)
                hog_feature = np.array(hog_features)[:,hog_win_start:hog_win_start+nblocks].ravel()
                features.append(hog_feature)

            features = np.concatenate(features)
            # 5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = clf.predict(test_features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                stripe_on_windows.append(window)
        result.append(stripe_on_windows)
    # 8) Return windows for positive detections
    return result

class Car():
    def __init__(self, bbox):
        self.x0 = x0(bbox)
        self.x1 = x1(bbox)
        self.bboxes = [bbox]
        self.boxwd = self.x1 - self.x0

class CarsDetector():
    def __init__(self):
        self.img_shape = None
        self.windows = None
        self.hot_wins = None
        self.height_shift = 0
        self.circle_shift = 0
        self.heatmap_array = None
        self.abs_heatmap = None
        self.cars = []
        self.window_stripes = []
        self.shifts = 6 # 5 recog only black car in test1 (overlap = (.9,.8))

    def find_hot_windows(self, img, model, overlap=(0.9, 0.8)):
        if self.windows==None:
            self.windows = slide_windows(img.shape, overlap=overlap, shifts=self.shifts, dbg=True)
        self.circle_shift = (self.circle_shift +1)%self.shifts
        self.height_shift = (self.height_shift +1)%len(self.windows[0])
        self.hot_wins = hot_windows(img, 
            self.windows[self.circle_shift], 
            model.classifier, 
            model.X_scaler, 
            **model.defaults)

    def analyze_current_stripe(self, process_image):
        for stripe in self.hot_wins:
            heatmap = npu.add_heat(process_image, bboxes=stripe)
            self.window_stripes = self.window_stripes + fe.labeled_heat_bboxes(heatmap, 2)

        hot_wins = self.window_stripes
        self.cars = sorted(self.cars, key=lambda car: car.boxwd, reverse=True)
        for car in self.cars:
            # if car.window:
            for i in range(len(hot_wins))[::-1]:
                if (car.x0 <= x1(hot_wins[i])) and\
                   (car.x1 >= x0(hot_wins[i])) and\
                   ((car.x1 - car.x0)*1.5 > (x1(hot_wins[i])-x0(hot_wins[i]))):
                    car.bboxes.append(hot_wins[i])
                    hot_wins.pop(i)
        if hot_wins:
            cars_found = []
            for win in hot_wins:
                car_found = True
                for car in cars_found:
                # loop won't run 1st time as cars_found is empty, but will next time as list is filled below 
                    if car.x0 <= x1(win) and car.x1 >= x0(win):
                        if (car.x1 - car.x0)*2 > x1(win) - x0(win):
                            car.bboxes.append(win)
                            car_found = False
                        break
                if car_found:
                    cars_found.append(Car(win))
            self.cars.extend(cars_found)

    def label_cars(self, process_image):
        outimg = np.copy(process_image)
        ghost_cars = []
        for icar in range(len(self.cars)):
            for i in range(1):
                if not self.cars[icar].bboxes:
                    break
                self.cars[icar].bboxes.pop(0)
            if not self.cars[icar].bboxes:
                ghost_cars.append(icar)
        for ghost in ghost_cars[::-1]:
            #print(ghost,ghost_cars,self.cars)
            self.cars.pop(ghost)

        for icar in range(len(self.cars)):
            while len(self.cars[icar].bboxes) > 40:
                self.cars[icar].bboxes.pop(0)
            if len(self.cars[icar].bboxes) > 12:
                heatmap = npu.add_heat(process_image, bboxes=self.cars[icar].bboxes)
                label_windows = fe.labeled_heat_bboxes(heatmap, 5)
                if label_windows:
                    self.cars[icar].window = label_windows[0]
                if self.cars[icar].boxwd > 20:
                    outimg = cv2.rectangle(outimg, self.cars[icar].window[0], 
                        self.cars[icar].window[1], (0,255,0), 2)
        return outimg

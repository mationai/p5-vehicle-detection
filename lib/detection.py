from types import SimpleNamespace as SNS
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lib import np_util as npu
from lib import feature_extraction as fe
from lib.helpers import *
from toolbox.draw_on_image import *

def horizontal_windows(img_shape, xmin=0, xmax=0, ymin=0, ymax=0,
                 winwd=64, winht=64, xyoverlap=(0.5, 0.5)):
    ''' Returns list of window coords to be slide across image horizontally
    '''
    xmax = xmax or img_shape[1]
    ymax = ymax or img_shape[0]
    pxs_per_xstep = int(winwd * (1-xyoverlap[0]))
    pxs_per_ystep = int(winht * (1-xyoverlap[1]))

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

def _calc_width(y, ymin=420, ymax=720, btm_wd=360, top_wd=32):
    # print( '%.2f'%((y-ymin)/(ymax-ymin)), (btm_wd-top_wd) , top_wd, int((y-ymin)/(ymax-ymin) * (btm_wd-top_wd) + top_wd))
    return int((y-ymin)/(ymax-ymin) * (btm_wd-top_wd) + top_wd)

def slide_windows(img_shape, ymin=440, ymax=None, maxht=.5, xyoverlap=(0, 0.8), shifts=9):
    imght, imgwd = img_shape[:2]
    max_winwd = int(maxht*imght) if 0<=maxht<=1 else int(maxht) 
    # ymin = int(ymin_pct*imght) recog only 1 car in test1 for pct=.61 or .62
    ymax = ymax or imght
    shifted_collection = []

    for shift in range(shifts):
        y = ymax
        x = 0
        winwd = max_winwd
        windows_collection = []
        i = 0

        while (winwd >= 64):
            # print(winwd, y)
# wd, y
# 360 720
# 276 649
# 212 594
# 163 552
# 125 520
# 97 496
# 75 477
            dead_step_x = int(winwd * (1-xyoverlap[0]) * shift/shifts)
            dead_x_position = x + dead_step_x
            y_step = int(winwd * (1-xyoverlap[1]))

            windows = horizontal_windows(img_shape, 
                xmin=dead_x_position, xmax=imgwd-x, ymin=y-winwd, ymax=y, 
                winwd=winwd, winht=winwd, xyoverlap=xyoverlap)

            y -= y_step
            winwd = _calc_width(y, ymin, ymax, btm_wd=max_winwd, top_wd=32)
            xwidth= _calc_width(y, ymin, ymax, btm_wd=imgwd*15, top_wd=32*15)

            x = 0 if xwidth > imgwd else (imgwd - xwidth)//2

            windows_collection.append(windows)
            i += 1

        shifted_collection.append(windows_collection)
        # print(shifted_collection,'\n')
    return shifted_collection

def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    on_windows = []
    current_window_area = ((0,0),(0,0))
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
        on_windows.append(stripe_on_windows)
    # 8) Return windows for positive detections
    return on_windows

class Vehicle():
    def __init__(self, bbox):
        self.x0 = x0(bbox)
        self.x1 = x1(bbox)
        self.bboxes = [bbox]
        self.boxwd = self.x1 - self.x0

class Vehicle_Collection():
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
        self.shifts = 6 # 5 recog only black car in test1 (xyoverlap=(.9,.8))

    def find_hot_windows(self, img, model, xyoverlap=(0.9, 0.8)):
        if self.windows==None:
            self.windows = slide_windows(img.shape, xyoverlap=xyoverlap, shifts=self.shifts)
        self.circle_shift = (self.circle_shift +1)%self.shifts
        self.height_shift = (self.height_shift +1)%len(self.windows[0])
        self.hot_wins = search_windows(img, 
            self.windows[self.circle_shift], 
            model.classifier, 
            model.X_scaler, 
            **model.defaults)

    def analyze_current_stripe(self, process_image):
        for stripe in self.hot_wins:
            heatmap = npu.add_heat(process_image, bboxes=stripe)
            self.window_stripes = self.window_stripes + fe.labeled_heat_windows(heatmap, 2)

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
                    cars_found.append(Vehicle(win))
            self.cars.extend(cars_found)

    def identify_vehicles(self, process_image):
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
                label_windows = fe.labeled_heat_windows(heatmap, 5)
                if label_windows:
                    self.cars[icar].window = label_windows[0]
                if self.cars[icar].boxwd > 20:
                    outimg = cv2.rectangle(outimg, self.cars[icar].window[0], 
                        self.cars[icar].window[1], (0,255,0), 2)
        return outimg

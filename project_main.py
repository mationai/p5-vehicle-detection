import numpy as np
import cv2
import time
import pickle
from sklearn.externals import joblib

from lib.detection import *
from config import picklefile, defaults
from types import SimpleNamespace as SNS

from moviepy.editor import VideoFileClip


t = time.time()
model = SNS(
    classifier = joblib.load(picklefile.svc),
    X_scaler = joblib.load(picklefile.X_scaler),
    defaults = defaults,
)
detector = CarsDetector(model)

def process_image(img):
    detector.detect_cars(img)
    return detector.detected_image(img)
    
#image = cv2.imread('./test_images/test6.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#processed_image = process_image(image)
#processed_image = cv2.cvtColor(processed_image,cv2.COLOR_RGB2BGR)
#cv2.imshow('Resulting Image', processed_image)

#cv2.imwrite('../output_images/test2_applied_lane_lines.jpg', combo)

video_output = './outputvid.mp4'
clip1 = VideoFileClip('./test1.mp4')
# clip1 = VideoFileClip('./test2.mp4')
# clip1 = VideoFileClip('./projend.mp4')
# clip1 = VideoFileClip('./test_video.mp4')
# clip1 = VideoFileClip('./project_video.mp4')
#clip1 = VideoFileClip('../harder_challenge_video.mp4')

white_clip_1 = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip_1.write_videofile(video_output, audio=False)

t2 = time.time()
m, s = divmod(t2 - t, 60)
print("%d:%02d to process video" % (m, s))

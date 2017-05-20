import numpy as np
import cv2, time
from sklearn.externals import joblib

from lib.detection import *
from config import pklpath, default, defaults
from types import SimpleNamespace as SNS

from moviepy.editor import VideoFileClip


t = time.time()
model = SNS(
    classifier = joblib.load(pklpath.svc),
    scaler = joblib.load(pklpath.scaler),
    train_size = default.train_size,
    defaults = defaults,
)
detector = CarDetector(model, 1280)

def process_image(img):
    return detector.detected_image(img)

# run_video = False
# run_video = './test1.mp4'
# run_video = './prob6.mp4'  #ffmpeg -i project_video.mp4 -ss 00:00:13 -codec copy -t 6 prob6.mp4
# run_video = './prob9.mp4'  #ffmpeg -i project_video.mp4 -ss 00:00:10.8 -codec copy -t 9 prob9.mp4
run_video = './prob10.mp4' #ffmpeg -i project_video.mp4 -ss 00:00:22.8 -codec copy -t 10 prob10.mp4
# run_video = './project_video.mp4'
    
if run_video:
    video_output = './outputvid.mp4'
    clip1 = VideoFileClip(run_video)
    # clip1 = VideoFileClip('./projend8.mp4')
    #project_video is 50secs, cut secs at exact secs gives good outputs
    # clip1 = VideoFileClip('./test_video.mp4')
    #clip1 = VideoFileClip('../harder_challenge_video.mp4')

    white_clip_1 = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip_1.write_videofile(video_output, audio=False)

    t2 = time.time()
    m, s = divmod(t2 - t, 60)
    print("%d:%02d to process video" % (m, s))
else:
    image = cv2.imread('./test_images/test6.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = process_image(image)
    processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    cv2.imshow('Resulting Image', processed)
    cv2.imwrite('output_images/processed.jpg', processed)

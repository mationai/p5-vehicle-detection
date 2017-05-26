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
detector = CarDetector(model, (720,1280))

def process_image(img):
    return detector.detected_image(img)

video_in = False
# video_in = './test1.mp4'
# video_in = './test_video.mp4'
# video_in = './prob6.mp4'  #ffmpeg -i project_video.mp4 -ss 00:00:13 -codec copy -t 6 prob6.mp4
# video_in = './prob8.mp4'  #ffmpeg -i project_video.mp4 -ss 00:00:38.4 -codec copy -t 8 prob8.mp4
# video_in = './prob9.mp4'  #ffmpeg -i project_video.mp4 -ss 00:00:10.8 -codec copy -t 9 prob9.mp4
# video_in = './prob10.mp4' #ffmpeg -i project_video.mp4 -ss 00:00:22.8 -codec copy -t 10 prob10.mp4
# video_in = './project_video.mp4'
    
if video_in:
    video_out = './outputvid.mp4'
    clip = VideoFileClip(video_in)
    
    clipped = clip.fl_image(process_image)  # NOTE: this function expects color images!!
    clipped.write_videofile(video_out, audio=False)

    t2 = time.time()
    m, s = divmod(t2 - t, 60)
    print("%d:%02d to process video" % (m, s))
else:
    ## run process_image() on single test image
    image = cv2.imread('./test_images/test6.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = process_image(image)
    processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output_images/processed.jpg', processed)

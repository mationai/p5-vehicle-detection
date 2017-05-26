import time, random
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from config import pklpath, defaults, default, cars_imgspath, notcars_imgspath
from lib import feature_extraction as fe


save_model = False
test_prediction_from_saved_model = False

t = time.time()

## Get car and non-car images
cars = [img for img in cars_imgspath]
notcars = [img for img in notcars_imgspath]
car_features = fe.images_features(cars, **defaults)
notcar_features = fe.images_features(notcars, **defaults)

## Fit a per-column scaler
X = np.vstack((car_features, notcar_features)).astype(np.float64)
scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

## Split to train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    # scaled_X, y, test_size=0.2, random_state=np.random.randint(0,100))
    scaled_X, y, test_size=0.2, random_state=123)

print('Using:', 
    'Color Space: %s\n'%default.color_space,
    'Spatial Bins Size: %s\n'%(default.spatial_size,) if default.spatial_feat else 'Spatial Feat: No\n', 
    'Color Hist Bins: %d\n'%default.hist_bins if default.hist_feat else 'Color Hist Feat: No\n', 
    'HOG channels: %s, orientations: %d, pxs per cell/cells per blk: %d,%d\n'%(
     default.hog_channel, default.orient, default.pix_per_cell, default.cell_per_block)
)
print('Feature vector length:', len(X_train[0]))

## Train model
svc = LinearSVC() #SVC() takes MUCH longer
svc.fit(X_train, y_train)

t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')

## Test accuracy
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


if save_model:
    joblib.dump(scaler, pklpath.scaler)
    joblib.dump(svc, pklpath.svc)

if test_prediction_from_saved_model:
    n_predict = 10
    sample = random.sample(range(len(y_test)), n_predict)
    clf = joblib.load(pklpath.svc)
    print('SVC prediction:', clf.predict(X_test[sample]))
    print('Actual labels: ', y_test[sample])

import time
import pickle
import numpy as np
import random

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from config import pklpath, defaults, default, cars_imgspath, notcars_imgspath
from lib import feature_extraction as fe
#        hog:  acc: feats: .2sec(@1.2sec) test vid recog:
# all3    12 - .9885 10224  both cars
# hogonly 12 - .9952  7056  neither
# nohist  12 - .9955 10128  black car only
# all3    11 - .9868  9636
# all3    10 - .989   9048

cars = [img for img in cars_imgspath]
notcars = [img for img in notcars_imgspath]

t = time.time()
car_features = fe.images_features(cars, **defaults)
notcar_features = fe.images_features(notcars, **defaults)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

X_train, X_test, y_train, y_test = train_test_split(
    # scaled_X, y, test_size=0.2, random_state=np.random.randint(0,100))
    scaled_X, y, test_size=0.2, random_state=123)

print('Using:', default.orient, 'orientations', default.pix_per_cell,
      'pixels per cell and', default.cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

svc = LinearSVC()
svc.fit(X_train, y_train)

t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')

# Test accuracy
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

joblib.dump(scaler, pklpath.scaler)
joblib.dump(svc, pklpath.svc)

# Test prediction from saved pkl file
n_predict = 10
sample = random.sample(range(len(y_test)), n_predict)
clf = joblib.load(pklpath.svc)
print('SVC prediction:', clf.predict(X_test[sample]))
print('Actual labels: ', y_test[sample])
# with open(picklefile.svc, 'wb') as file:
#     pickle.dump(svc, file, protocol=pickle.HIGHEST_PROTOCOL)
# with open(picklefile.X_scaler, 'wb') as file:
#     pickle.dump(X_scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

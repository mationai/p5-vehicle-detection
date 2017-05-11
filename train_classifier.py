import cv2
import glob
import time
import pickle
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from config import picklefile, defaults, default, cars_imgspath, notcars_imgspath

from lib import feature_extraction as fe
#           secs: acc: feats: .2sec(@1.2sec) test vid recog:
# all3     - 149 .9958 10224  both cars
# hogonly  - 111 .9952  7056  neither
# nohist   - 133 .9955 10128  black car only

def train_classifier(svc_pickle='./detection_functions/svc_pickle.pickle', X_scaler_pickle='./detection_functions/X_scaler_pickle.pickle'):
    # Read in notcars
    cars = [img for img in cars_imgspath]
    notcars = [img for img in notcars_imgspath]

    t = time.time()
    car_features = fe.images_features(cars, **defaults)
    notcar_features = fe.images_features(notcars, **defaults)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', default.orient, 'orientations', default.pix_per_cell,
          'pixels per cell and', default.cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # svc = SVC()
    # Check the training time for the SVC
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample

    classifier_values = {'svc': svc, 'X_scaler': X_scaler}
    with open(picklefile.svc, 'wb') as file:
        pickle.dump(svc, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(picklefile.X_scaler, 'wb') as file:
        pickle.dump(X_scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    train_classifier()
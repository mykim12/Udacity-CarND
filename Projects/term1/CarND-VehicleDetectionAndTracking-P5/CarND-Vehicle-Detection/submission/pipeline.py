import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pylab import savefig

import numpy as np
import time
import cv2
import glob
from skimage.feature import hog
#from skimage import color, exposure
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from random import shuffle, randint
import pickle
from scipy.ndimage.measurements import label

class Data(object):
    def __init__(self, _dpath="../data"):
        self.data_path = _dpath

        self.cars = []
        self.non_cars = []
        self.data_dict = {}

    #------------------------------------------------------------
    # DEF load_data
    # - load data from specified data directory
    # - if _sample is specified, only collect the size of _sample
    #   (used for debugging, and speed up test runs)
    #------------------------------------------------------------
    def load_data(self, _car_sample=0, _noncar_sample=0):
        self.cars = []
        self.non_cars = []
        self.data_dict = {}
        
        image_files = list(glob.iglob(self.data_path + "/**/**/*.png", recursive=True))
        #print("Total %d image files found"%len(image_files))

        # shuffle
        shuffle(image_files)

        for imgf in image_files:
            if 'GTI' not in imgf:   continue

            if 'non-' in imgf:
                if (_noncar_sample > 0) and (len(self.non_cars) >= _noncar_sample):
                    continue
                self.non_cars.append(imgf)

            else:
                if (_car_sample > 0) and (len(self.cars) >= _car_sample):
                    continue
                self.cars.append(imgf)

        self.data_dict["n_cars"] = len(self.cars)
        self.data_dict["n_non_cars"] = len(self.non_cars)

        #print("len(self.cars): ", len(self.cars))
        sample_img = mpimg.imread(self.cars[0])
        self.data_dict["image_shape"] = sample_img.shape
        self.data_dict["data_type"] = sample_img.dtype

        print("Total %d cars and %d non-cars of size"%\
              (self.data_dict["n_cars"], self.data_dict["n_non_cars"]), \
              self.data_dict["image_shape"], \
              "and data type:", self.data_dict["data_type"], \
              "are collected from data directory %s"%(self.data_path))

    #------------------------------------------------------------
    # DEF visualizeSample
    # - visualize a sample car & non_car image
    #------------------------------------------------------------
    def visualizeSample(self):
        # randomly select indices
        car_ind, non_car_ind = self.pickRandomImages()

        # Read in car / not-car images
        car_image = mpimg.imread(self.cars[car_ind])
        non_car_image = mpimg.imread(self.non_cars[non_car_ind])

        # Plot the examples
        plt.ion()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))
        #f.tight_layout()
        ax1.imshow(car_image)
        ax1.set_title('Example Car Image')
        ax2.imshow(non_car_image)
        ax2.set_title('Example Non-car Image')
        plt.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.05)
        input("showing sample car & non-car images![Enter]")
        plt.ioff()
        plt.savefig("./output_images/car_and_not_car_sample.png")

    #------------------------------------------------------------
    # DEF pickRandomImages
    # - random sampling
    #------------------------------------------------------------
    def pickRandomImages(self):
        car_ind = np.random.randint(0, len(self.cars))
        non_car_ind = np.random.randint(0, len(self.non_cars))

        return car_ind, non_car_ind


class Features(object):
    def __init__(self, _data):
        self.data = _data

    #------------------------------------------------------------
    # DEF setHOGParams
    # - configure HOG parameters
    #------------------------------------------------------------
    def setHOGParams(self, _ppc=8, _cpb=2, _orient=9, _vis=True, _chnl='ALL', _fv=False):
        self.pix_per_cell = _ppc
        self.cell_per_block = _cpb
        self.orient = _orient
        self.vis = _vis
        self.hog_channel = _chnl    # Can be 0, 1, 2, or "ALL"
        self.featvec = _fv

    #------------------------------------------------------------
    # DEF HOG_expr
    # - some HOG experiments
    #------------------------------------------------------------
    def HOG_expr(self):
        # pick a random set of showSamples
        car_ind, non_car_ind = self.data.pickRandomImages()
        # read in
        car = mpimg.imread(self.data.cars[car_ind])
        car_gray = cv2.cvtColor(car, cv2.COLOR_RGB2GRAY)
        non_car = mpimg.imread(self.data.non_cars[non_car_ind])
        non_car_gray = cv2.cvtColor(non_car, cv2.COLOR_RGB2GRAY)

        if self.vis:
            car_features, car_hog_image = self.HOG(car_gray)
            non_car_features, non_car_hog_image = self.HOG(non_car_gray)

            # Plot the examples
            plt.ion()
            f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
            ax1.imshow(car)
            ax1.set_title("Car")
            ax2.imshow(car_gray, cmap='gray')
            ax2.set_title("Car Gray")
            ax3.imshow(car_hog_image, cmap='gray')
            ax3.set_title('Car HOG')
            ax4.imshow(non_car)
            ax4.set_title('Non-car')
            ax5.imshow(non_car_gray, cmap='gray')
            ax5.set_title('Non-car Gray')
            ax6.imshow(non_car_hog_image, cmap='gray')
            ax6.set_title('Non-car HOG')

            plt.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.05)
            input("showing sample car & non-car HOG images![Enter]")
            plt.ioff()
            plt.savefig("./output_images/HOG_sample_1.png")

        else:
            car_features = self.HOG(car_gray)
            non_car_features = self.HOG(non_car_gray)

    #------------------------------------------------------------
    # DEF HOG_expr2
    # - some HOG experiments
    #------------------------------------------------------------
    def HOG_expr2(self):
        # pick a random set of showSamples
        car_ind, non_car_ind = self.data.pickRandomImages()
        # read in
        car = mpimg.imread(self.data.cars[car_ind])
        car_gray = cv2.cvtColor(car, cv2.COLOR_RGB2GRAY)

        # cpb change
        self.setHOGParams(_ppc=8, _cpb=2, _orient=9, _vis=True)
        f1, i1 = self.HOG(car_gray)
        self.setHOGParams(_ppc=8, _cpb=4, _orient=9, _vis=True)
        f2, i2 = self.HOG(car_gray)
        self.setHOGParams(_ppc=8, _cpb=6, _orient=9, _vis=True)
        f3, i3 = self.HOG(car_gray)

        # ppc change
        self.setHOGParams(_ppc=4, _cpb=2, _orient=9, _vis=True)
        f4, i4 = self.HOG(car_gray)
        self.setHOGParams(_ppc=16, _cpb=2, _orient=9, _vis=True)
        f5, i5 = self.HOG(car_gray)

        # orient change
        self.setHOGParams(_ppc=8, _cpb=2, _orient=4, _vis=True)
        f6, i6 = self.HOG(car_gray)
        self.setHOGParams(_ppc=8, _cpb=2, _orient=20, _vis=True)
        f7, i7 = self.HOG(car_gray)

        # Plot the examples
        plt.ion()
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15, 15))

        # cpb expr
        ax1.imshow(i1, cmap='gray')
        ax1.set_title("Original (ppc=8, cpb=2, orient=9)")
        ax2.imshow(i2, cmap='gray')
        ax2.set_title("cpb expr (ppc=8, cpb=4, orient=9)")
        ax3.imshow(i3, cmap='gray')
        ax3.set_title("cpb expr (ppc=8, cpb=6, orient=9)")

        # ppc expr
        ax4.imshow(i1, cmap='gray')
        ax4.set_title("Original (ppc=8, cpb=2, orient=9)")
        ax5.imshow(i4, cmap='gray')
        ax5.set_title("ppc expr (ppc=4, cpb=2, orient=9)")
        ax6.imshow(i5, cmap='gray')
        ax6.set_title("ppc expr (ppc=16, cpb=2, orient=9)")

        # orient expr
        ax7.imshow(i1, cmap='gray')
        ax7.set_title("Original (ppc=8, cpb=2, orient=9)")
        ax8.imshow(i6, cmap='gray')
        ax8.set_title("orient expr (ppc=8, cpb=2, orient=4)")
        ax9.imshow(i7, cmap='gray')
        ax9.set_title("orient expr (ppc=8, cpb=2, orient=20)")

        plt.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.05)
        input("showing sample car & non-car HOG images![Enter]")
        plt.ioff()
        plt.savefig("./output_images/HOG_sample_2.png")
        mpimg.imsave("./output_images/HOG_sample_2_origimg.png", car)



    #------------------------------------------------------------
    # DEF HOG
    # - hog wrapper
    #------------------------------------------------------------
    def HOG(self, _img):
        return hog(_img, orientations=self.orient,
                         pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                         cells_per_block=(self.cell_per_block, self.cell_per_block),
                         transform_sqrt=False,
                         visualise=self.vis,
                         feature_vector=self.featvec)

    #------------------------------------------------------------
    # DEF bin_spatial
    # - compute binned color features
    #------------------------------------------------------------
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
#        features = cv2.resize(img, size).ravel()

        color1 = cv2.resize(img[:,:,0], size).ravel()
        color2 = cv2.resize(img[:,:,1], size).ravel()
        color3 = cv2.resize(img[:,:,2], size).ravel()

        features = np.hstack((color1, color2, color3))
        # Return the feature vector
        return features


    #------------------------------------------------------------
    # DEF color_hist
    # - compute color histogram features
    #------------------------------------------------------------
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    #------------------------------------------------------------
    # DEF setFeatParams
    # - configure other feature parameters
    #   (for spatial and histogram features)
    #------------------------------------------------------------
    def setFeatParams(self, _cspace='RGB', _ssize=(32, 32), _hist_bins=32, _hist_range=(0, 256)):
        self.cspace = _cspace
        self.spatial_size = _ssize
        self.hist_bins = _hist_bins
        self.hist_range = _hist_range

    #------------------------------------------------------------
    # DEF extractFeatures
    # - extract spatial and histogram features
    #------------------------------------------------------------
    def extractFeatures(self, imgs, _cf=True, _hog=False, _imgbuf=False):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        # Read in each one by one
        for item in imgs:
            if _imgbuf:
                image = item
                feature_image = np.copy(image)
            else:
                image = mpimg.imread(item)
                # apply color conversion if other than 'RGB'
                if self.cspace != 'RGB':
                    if self.cspace == 'HSV':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    elif self.cspace == 'LUV':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                    elif self.cspace == 'HLS':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                    elif self.cspace == 'YUV':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                    elif self.cspace == 'YCrCb':
                        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

                else: feature_image = np.copy(image)

            if _cf:
                # Apply bin_spatial() to get spatial color features
                spatial_features = self.bin_spatial(feature_image, size=self.spatial_size)
                # Apply color_hist() also with a color space option now
                hist_features = self.color_hist(feature_image, nbins=self.hist_bins, bins_range=self.hist_range)
            if _hog:
                # Get HOG features
                if self.hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.HOG(feature_image[:, :, channel]))

                    hog_features = np.ravel(hog_features)
                else:
                    #hog_features = self.HOG(feature_image[:,:,self.hog_channel])
                    if _imgbuf:
                        hog_features = self.HOG(feature_image)
                    else:
                        hog_features = self.HOG(feature_image[:, :, self.hog_channel])


            # Append the new feature vector to the features list
            if _cf and _hog:
                features.append(np.concatenate((spatial_features, hist_features, hog_features)))
            elif _cf:
                features.append(np.concatenate((spatial_features, hist_features)))
            else:
                features.append(hog_features)


        #print("features:", len(features[0]))
        # Return list of feature vectors
        if len(features) == 1:
            return features[0]
        return features


    #------------------------------------------------------------
    # DEF extractFeaturesNorm
    # - extract spatial and histogram features, and NORMALIZE!
    #------------------------------------------------------------
    def extractFeaturesNorm(self, _cf=True, _hog=False, _singleImg=None):
        if _singleImg is not None:
            #print("============ singleImg ======================")
            self.features = self.extractFeatures([_singleImg], _cf, _hog, _imgbuf=True)
            if len(self.features) > 0:
                #self.X_scaler = StandardScaler().fit(np.array(self.features).reshape(1, -1).astype(np.float64))
                self.singleImgFeatures = self.features
                #self.singleImgFeatures = self.X_scaler.transform(np.array(self.features).reshape(1, -1))
            else:
                print('Your function only returns empty feature vectors...')

        else:
            self.car_features = self.extractFeatures(self.data.cars, _cf, _hog)
            self.non_car_features = self.extractFeatures(self.data.non_cars, _cf, _hog)
            if len(self.car_features) > 0:
                # Create an array stack of feature vectors
                self.X = np.vstack((self.car_features, self.non_car_features)).astype(np.float64)
                # Fit a per-column scaler
                self.X_scaler = StandardScaler().fit(self.X)
                # Apply the scaler to X
                self.scaled_X = self.X_scaler.transform(self.X)

            else:
                print('Your function only returns empty feature vectors...')

    #------------------------------------------------------------
    # DEF extractFeatures_visualize
    # - visualization of results from extractFeatures_expr()
    #------------------------------------------------------------
    def extractFeatures_visualize(self):
        # visualize
        car_ind = np.random.randint(0, len(self.data.cars))
        # Plot an example of raw and scaled features
        plt.ion()
        fig = plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(self.data.cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(self.X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(self.scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        input("Showing sample extracted feature result![Enter]")
        plt.ioff()
        plt.savefig("./output_images/final_features_norm.png")



class Classifier(object):
    def __init__(self, _FT, _testSize=0.2, _modelFile='model.p'):
        self.Feature = _FT
        self.tsize = _testSize
        self.modelFile = _modelFile

    def SVM(self):
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        svr = svm.SVC()
        clf = grid_search.GridSearchCV(svr, parameters)
        clf.fit(iris.data, iris.target)

    #------------------------------------------------------------
    # DEF SVC_ColorFeatures
    # - SVM Classifier using Color features
    #------------------------------------------------------------
    def SVC_ColorFeatures(self, _npredict=10, _img=None):
        #print("======== SVC_ColorFeatures() ========")
        self.Feature.extractFeaturesNorm(_cf=True, _hog=False, _singleImg=_img)
        if _img is None:
            self.train(_npredict)
        else:
            return self.Feature.singleImgFeatures


    #------------------------------------------------------------
    # DEF SVC_OGFeatures
    # - SVM Classifier using HOG features
    #------------------------------------------------------------
    def SVC_HOGFeatures(self, _npredict=10, _img=None, _channel=0):
        #print("======== SVC_HOGFeatures() ========")
        self.Feature.hog_channel = _channel
        self.Feature.extractFeaturesNorm(_cf=False, _hog=True, _singleImg=_img)
        if _img is None:
            self.train(_npredict)
        else:
            return self.Feature.singleImgFeatures


    #------------------------------------------------------------
    # DEF SVC_AllFeatures
    # - SVM Classifier using ALL features
    #------------------------------------------------------------
    def SVC_AllFeatures(self, _npredict=10, _img=None):
        #print("======== SVC_AllFeatures() ========")
        self.Feature.extractFeaturesNorm(_cf=True, _hog=True, _singleImg=_img)
        if _img is None:
            return self.train(_npredict)
        else:
            return self.Feature.singleImgFeatures



    #------------------------------------------------------------
    # DEF train
    # - train entry point
    #------------------------------------------------------------
    def train(self, _npredict):
        # Define the labels vector
        y = np.hstack((np.ones(len(self.Feature.car_features)),
                       np.zeros(len(self.Feature.non_car_features))))
        #print("y.shape: ", y.shape)

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            self.Feature.scaled_X, y, test_size=self.tsize, random_state=rand_state)

        print('Using spatial binning of:',self.Feature.spatial_size,
            'and', self.Feature.hist_bins,'histogram bins')
        print('Feature vector length:', len(X_train[0]))

        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        accuracy = round(self.svc.score(X_test, y_test), 4)
        print('Test Accuracy of SVC = ', accuracy)
        # Check the prediction time for a single sample
        t=time.time()
        print('My SVC predicts: ', self.svc.predict(X_test[0:_npredict]))
        print('For these',_npredict, 'labels: ', y_test[0:_npredict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', _npredict, 'labels with SVC')

        return accuracy


    #------------------------------------------------------------
    # DEF saveModel
    # - save trained svc
    #------------------------------------------------------------
    def saveModel(self, _f=None):
        modelf = self.modelFile
        if _f is not None:
            modelf = _f
        print("saving trained svc to %s"%modelf)
        model_dict = {"svc":self.svc, "X_scaler":self.Feature.X_scaler}
        pickle.dump(model_dict, open(modelf, 'wb'))



    #------------------------------------------------------------
    # DEF loadModel
    # - load trained svc
    #------------------------------------------------------------
    def loadModel(self, _f=None):
        modelf = self.modelFile
        if _f is not None:
            modelf = _f
        print("loading trained svc from %s"%modelf)
        model_dict = pickle.load(open(modelf, 'rb'))
        self.svc = model_dict["svc"]
        self.Feature.X_scaler = model_dict["X_scaler"]
 


class Detector(object):
    def __init__(self, _classifier, _yss=[None, None],
                       _xywin=(96, 96), _xyoverlap=(0.5, 0.5)):
        self.classifier = _classifier
        self.y_start_stop = _yss
        self.xy_window = _xywin
        self.xy_overlap = _xyoverlap


    def setYSS(self, _yss):
        self.y_start_stop = _yss

    #------------------------------------------------------------
    # DEF slideWindow
    # - find windows in sliding fashion
    #------------------------------------------------------------
    def slideWindow(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        self.windows = []

        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
        # Initialize a list to append window positions to

        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                self.windows.append(((startx, starty), (endx, endy)))

    #------------------------------------------------------------
    # DEF searchWindows
    # - search windows which has promising features
    #------------------------------------------------------------
    def searchWindows(self, _img):
        #1) Create an empty list to receive positive detection windows
        on_windows = []

        #2) Iterate over all windows in the list
        for window in self.windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(_img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

            #4) Extract features for that window using single_img_features()
            self.classifier.Feature.setHOGParams(_ppc=8, _cpb=2, _orient=9, _vis=False, _chnl='ALL', _fv=True)
            all_features = self.classifier.SVC_AllFeatures(_img=test_img)

            # scale
            test_features = self.classifier.Feature.X_scaler.transform(all_features)

            #5) Predict
            prediction = self.classifier.svc.predict(test_features)
            print("prediction: ", prediction)

            #6) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #7) Return windows for positive detections
        return on_windows


    #------------------------------------------------------------
    # DEF detect
    # - detect promising resions of windows with an image
    #------------------------------------------------------------
    def detect(self, _img):
        self.image = _img
        self.slideWindow(image, x_start_stop=[None, None], y_start_stop=self.y_start_stop,
                         xy_window=self.xy_window, xy_overlap=self.xy_overlap)
        self.hot_windows = self.searchWindows(image)
        #print("len(hot_windows):", self.hot_windows)

    def visualize(self):
        draw_image = np.copy(self.image)
        window_img = self.drawBoxes(draw_image, self.hot_windows, _color=(0, 0, 255), _thick=6)

        plt.ion()
        plt.imshow(window_img)
        input("showing sample detection result!")
        plt.ioff()
        mpimg.imsave("./output_images/detectionWindows_4.png", window_img)

    def drawBoxes(self, _img, _bboxes, _color, _thick):
        # Make a copy of the image
        imcopy = np.copy(_img)
        # Iterate through the bounding boxes
        for bbox in _bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], _color, _thick)
        # Return the image copy with boxes drawn
        return imcopy

    #------------------------------------------------------------
    # DEF convertColor
    # - convert colorspace
    #------------------------------------------------------------
    def convertColor(self, img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

    #------------------------------------------------------------
    # DEF findCars
    # - extract features using hog sub-sampling and make predictions
    #------------------------------------------------------------
    def findCars(self, _img, _scale):
        print("------------ findCars() -------------")
        self.image = _img
        draw_img = np.copy(self.image)
        self.image = self.image.astype(np.float32)/255
        #self.image = self.image.astype(np.float32)
        boxes = []

        FEAT = self.classifier.Feature
        scale = _scale
        orient = FEAT.orient
        pix_per_cell = FEAT.pix_per_cell
        cell_per_block = FEAT.cell_per_block
        X_scaler = FEAT.X_scaler
        spatial_size = FEAT.spatial_size
        hist_bins = FEAT.hist_bins
        hist_range = FEAT.hist_range
        svc = self.classifier.svc

        ystart, ystop = self.y_start_stop
        img_tosearch = self.image[ystart:ystop,:,:]
        ctrans_tosearch = self.convertColor(img_tosearch, conv='RGB2YCrCb')

        num_positives = 0
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0].copy()
        ch2 = ctrans_tosearch[:,:,1].copy()
        ch3 = ctrans_tosearch[:,:,2].copy()

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1
        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        #window = cell_per_block * pix_per_cell
        nblocks_per_window = (window // pix_per_cell)-1
        #print("nblocks_per_window: ", nblocks_per_window)
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = self.classifier.SVC_HOGFeatures(_img=ch1, _channel=0)
        hog2 = self.classifier.SVC_HOGFeatures(_img=ch2, _channel=1)
        hog3 = self.classifier.SVC_HOGFeatures(_img=ch3, _channel=2)
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                # Extract the image patch
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                '''
                spatial_features = FEAT.bin_spatial(subimg, size=spatial_size)
                hist_features = FEAT.color_hist(subimg, nbins=hist_bins, bins_range=hist_range)
                print("len(spatial_feature): ", len(spatial_features))
                print("len(hist_feature): ", len(hist_features))
                '''
                color_features = self.classifier.SVC_ColorFeatures(_img=subimg)

                # Scale features and make a prediction
#                all_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                all_features = np.hstack((color_features, hog_features)).reshape(1, -1)

                # scale
                test_features = X_scaler.transform(all_features)
                #test_features = X_scaler.transform(all_features2)

                # predict
                test_prediction = self.classifier.svc.predict(test_features)

                # draw rectangle if predicted true (car)
                if test_prediction == 1:
                    num_positives += 1
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    boxes.append([(xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)])
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

#        print("num_positives: ", num_positives)
#        plt.ion()
#        plt.imshow(draw_img)
#        input("showing sample detection result!")
#        plt.ioff()
#        mpimg.imsave("./output_images/detection_findCars_2.png", draw_img)

        return boxes


    #------------------------------------------------------------
    # DEF add_heat
    #------------------------------------------------------------
    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1


        # Return updated heatmap
        return heatmap# Iterate through list of bboxes


    #------------------------------------------------------------
    # DEF apply_threshold
    #------------------------------------------------------------
    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap


    #------------------------------------------------------------
    # DEF draw_labeled_bboxes
    #------------------------------------------------------------
    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # filter out obvious false positives using bbox ratio
            bbox_w = np.max(nonzerox) - np.min(nonzerox)
            bbox_h = np.max(nonzeroy) - np.min(nonzeroy)

            area = bbox_w * bbox_h
            ratio = float(bbox_h) / float(bbox_w)
            if area < 3000: continue
            if area > 70000: continue

            if ratio < 0.3: continue
            if ratio > 1.9: continue

#            print("area: ", area, "ratio: ", ratio)
            
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img



    #------------------------------------------------------------
    # DEF heatMap
    # - draw some heatmaps
    #------------------------------------------------------------
    def heatMap(self, _img, _box_list, _thr=1, _vis=False, _idx=0):
        image = _img

        heat = np.zeros_like(image[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        heat = self.add_heat(heat, _box_list)

        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat, _thr)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(np.copy(image), labels)
        print(labels[1], "cars found.")

        if _vis:
            plt.ion()
            fig = plt.figure(figsize=(12,4))

            ax = fig.add_subplot(131)
            ax.imshow(heatmap, cmap='hot')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title('Heat Map')

            ax2 = fig.add_subplot(132)
            ax2.imshow(labels[0], cmap='gray')
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_title('after label')

            ax3 = fig.add_subplot(133)
            ax3.imshow(draw_img)
            ax3.set_xticklabels([])
            ax3.set_yticklabels([])
            ax3.set_title('Car Positions - (%d) cars'%labels[1])

            fig.tight_layout()
            plt.ioff()
            savefig("./output_images/detection_heatMap_figure_A_%d.png"%_idx, bbox_inches='tight')

        return draw_img



    #------------------------------------------------------------
    # DEF heatMap_acc
    # - draw some heatmaps (accumulated)
    #------------------------------------------------------------
    def heatMap_acc(self, _gidx, _acc, _thr=[1, 3], _vis=False, _save='vid_frames'):
        start_frame = _gidx - len(_acc) + 1
        print("start_frame: ", start_frame)

        images = []
        images_draw = []
        heatmaps = []
        all_boxes = []
        for item in _acc:
            image, boxes = item
            images.append(image)

            all_boxes += boxes

            heat = np.zeros_like(image[:,:,0]).astype(np.float)
            heat = self.add_heat(heat, boxes)
            heat = self.apply_threshold(heat, _thr[0])
            heatmap = np.clip(heat, 0, 255)

            draw_image = np.copy(image)
            d_img = self.drawBoxes(draw_image, boxes, _color=(0, 0, 255), _thick=4)

            images_draw.append(d_img)
            heatmaps.append(heatmap)


        if _vis:
            plt.ion()
            #fig = plt.figure(figsize=(12,4))
            fig, ax = plt.subplots(6, 2, figsize=(4, 9))
            for i in range(6):
                ax[i][0].imshow(images_draw[i])
                ax[i][0].set_xticklabels([])
                ax[i][0].set_yticklabels([])
                ax[i][0].set_title('Image %05d'%(start_frame + i))

                ax[i][1].imshow(heatmaps[i], cmap='hot')
                ax[i][1].set_xticklabels([])
                ax[i][1].set_yticklabels([])
                ax[i][1].set_title('Image %05d'%(start_frame + i))

            fig.tight_layout()
            plt.ioff()
            savefig("./output_images/detection_heatMap_figure_Acc_%05d.png"%start_frame, bbox_inches='tight')

            a = input("here")


        # generate accumulated heatmap
        g_heat = np.zeros_like(_acc[0][0][:,:,0]).astype(np.float)
        g_heat = self.add_heat(heat, all_boxes)
        g_heat = self.apply_threshold(g_heat, _thr[1])
        g_heatmap = np.clip(g_heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(g_heatmap)

        # draw on all images
        for idx, img in enumerate(images):
            draw_img = self.draw_labeled_bboxes(np.copy(img), labels)
#            plt.ion()
#            plt.imshow(draw_img)
#            plt.ioff()
#            a = input("asdf")
            imgpath = "./%s/frame_%05d.jpg"%(_save, start_frame + idx)
            print("imgpath: ", imgpath)
            mpimg.imsave(imgpath, draw_img)

        


#------------------------------------------------------------
# MAIN ENTRY POINT
#------------------------------------------------------------
if __name__ == "__main__":
    #------------------------------------------------------------
    # STEP 1. HOG (Histogram of Oriented Gradients)
    #------------------------------------------------------------
    myData = Data("../data")
    myData.load_data(_car_sample=3000, _noncar_sample=3000)
    #myData.visualizeSample()

    #------------------------------------------------------------
    # STEP 2. FEATURE EXTRACTION
    #------------------------------------------------------------
    FEAT = Features(myData)
    #FEAT.setHOGParams(_ppc=8, _cpb=2, _orient=9, _vis=True)
    #FEAT.HOG_expr()
    #FEAT.HOG_expr2()

    #FEAT.setFeatParams(_cspace='RGB', _ssize=(32, 32),
    #                   _hist_bins=32, _hist_range=(0, 256))
    #FEAT.setHOGParams(_ppc=16, _cpb=2, _orient=9, _vis=False)
    #FEAT.extractFeaturesNorm()
    #FEAT.extractFeatures_visualize()

    #------------------------------------------------------------
    # STEP 3. CLASSIFIER
    #------------------------------------------------------------
    CLSF = Classifier(FEAT, _testSize=0.3, _modelFile="model2.p")

    # train with color features
    #CLSF.Feature.setFeatParams(_cspace='RGB', _ssize=(32, 32),
    CLSF.Feature.setFeatParams(_cspace='YCrCb', _ssize=(32, 32),
                               _hist_bins=32, _hist_range=(0, 256))
    #CLSF.SVC_ColorFeatures(_npredict=100)

    # train with HOG features
    #CLSF.Feature.setHOGParams(_ppc=8, _cpb=2, _orient=9, _vis=False, _chnl=0, _fv=True)
    #CLSF.SVC_HOGFeatures(_npredict=100)

    # train with all features (color & HOG)
    CLSF.Feature.setHOGParams(_ppc=8, _cpb=2, _orient=9, _vis=False, _chnl='ALL', _fv=True)

    acc = 0.0
    # desired accuracy
    max_acc = 0.997
    acc_thr = 0.992
    trial = 0
#    while(acc < max_acc):
#        trial += 1
#        print("\n[ Iter %d ]"%trial)
#        # randomly generate training data with random numbers
#        CLSF.Feature.data.load_data(_car_sample=randint(800, 2800), _noncar_sample=randint(800, 3800))
#        acc = CLSF.SVC_AllFeatures(_npredict=-1)
#        if acc > acc_thr:
#            CLSF.saveModel("model_n2_acc_%.4f.p"%acc)
#
#    a = input("aa")

    # save model
    #CLSF.saveModel()

    # load the model from file
    #CLSF.loadModel('../models/model_n2_acc_0.9930.p')
    CLSF.loadModel('../models/model_n2_acc_0.9936.p_BEST')


    #------------------------------------------------------------
    # STEP 3. DETECTOR
    #------------------------------------------------------------
    DET = Detector(_classifier=CLSF, _yss=[200, 660])
    image = mpimg.imread('./test_images/test4.jpg')
    DET.classifier.Feature.setHOGParams(_ppc=8, _cpb=2, _orient=9, _vis=False, _chnl='ALL', _fv=False)
    #DET.detect(image)
    #DET.visualize()


    #------------------------------------------------------------
    # STEP 3.1. DETECTOR - findCar()
    #------------------------------------------------------------
    DET.setYSS((400, 656))
    # set default HOGParams
    # - later _chnl info will be updated when needed
    DET.classifier.Feature.setHOGParams(_ppc=8, _cpb=2, _orient=9, _vis=False, _chnl=1, _fv=False)
#    boxes = DET.findCars(image, _scale=1.2)
#    boxes1 = DET.findCars(image, _scale=1.5)
#    boxes2 = DET.findCars(image, _scale=1.8)
#
#    full_result = DET.drawBoxes(image, boxes+boxes1+boxes2, (0,0,255), _thick=6)
#    plt.ion()
#    plt.imshow(full_result)
#    input("showing sample full detection result!")
#    plt.ioff()
#    mpimg.imsave("./output_images/detection_findCars_3.png", full_result)
#
#    a = input("aa")

    #------------------------------------------------------------
    # STEP 4.1. Video Implementation - HeatMap
    #------------------------------------------------------------
    scales = [1.2]
#    boxes_all = []
#    for s in scales:
#        boxes = DET.findCars(image, _scale=s)
#        boxes_all += boxes
#    _ = DET.heatMap(image, boxes, 2, _vis=True)
#
#    a = input("aa")


    #------------------------------------------------------------
    # STEP 4.2. Video Implementation - Video Detection
    #------------------------------------------------------------
    videof = "project_video.mp4"
    save_path = "vid_frames_6"
    vidcap = cv2.VideoCapture(videof)
    success = True
    #threshold = float(len(boxes_all) * 0.08)
    threshold = 2
    g_threshold = 12
    count = 1
    accumulate_no = 6
    accumulated_result = []
    # 1029 - 1057
    while success:
        success, frame = vidcap.read()
        if not success:
            break

#        if count > 1057:
#            break
#
#        if count < 1027:
#            count += 1
#            continue

        print("[IDX: %d] Read a new frame: "%count)
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        boxes_all = []
        for s in scales:
            boxes = DET.findCars(image, _scale=s)
            boxes_all += boxes

        accumulated_result.append([image, boxes_all])

        if count % accumulate_no == 0:
            DET.heatMap_acc(count, accumulated_result, [threshold, g_threshold], _vis=False, _save=save_path)
            #DET.heatMap_acc(count, accumulated_result, [threshold, g_threshold], _vis=False)
            accumulated_result = []

        count += 1

        
        


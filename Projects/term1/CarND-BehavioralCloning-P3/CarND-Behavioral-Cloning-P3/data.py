import cv2
import numpy as np
import csv
import sklearn
from sklearn.model_selection import train_test_split



class DataSet(object):
    PROVIDED = 'provided'
    EXP1 = 'exp1'
    ALL = 'all'



class BCData(object):

    def __init__(self, _dataSet, _fitgen=False):
        self.data_path = "/data/udacity/bcloning_data"
        self.subset = _dataSet
        self.fit_gen = _fitgen

        self.augmented = True

        self.loadData()


    ##------------------------------
    # DEF loadData()
    # - load image and annotations
    #   from file
    #-------------------------------
    def loadData(self):
        if not self.fit_gen:
            images = []
            measurements = []

        samples = []
        csvfile = '/'.join([self.data_path, self.subset, 'driving_log.csv'])
        with open(csvfile) as fp:
            reader = csv.reader(fp)
            next(reader)
            for line in reader:
                samples.append(line)

                if not self.fit_gen:
                    fname = '/'.join([self.data_path, self.subset, "IMG", line[0].split('/')[-1]])
                    #i = cv2.imread(line[0])
                    i = cv2.imread(fname)
                    images.append(i)
                    m = float(line[3])
                    measurements.append(m)

        if not self.fit_gen:
            self.X_train = np.asarray(images)
            self.y_train = np.asarray(measurements)
            print("X_train.shape: ", self.X_train.shape)
            print("y_train.shape: ", self.y_train.shape)

            #('line: ', ['/media/mykim/SHRD_DATA/udacity/bcloning_data/exp4/IMG/center_2017_04_21_18_14_03_950.jpg', 
            #            '/media/mykim/SHRD_DATA/udacity/bcloning_data/exp4/IMG/left_2017_04_21_18_14_03_950.jpg', 
            #            '/media/mykim/SHRD_DATA/udacity/bcloning_data/exp4/IMG/right_2017_04_21_18_14_03_950.jpg', 
            #            '-0.05031446', '1', '0', '30.19084'])

        self.n_samples = len(samples)
        print("%d samples are collected"%self.n_samples)

        self.train_samples, self.validation_samples = train_test_split(samples, test_size=0.2)
        self.num_tr_samples = len(self.train_samples)
        self.num_val_samples = len(self.validation_samples)


    ##------------------------------
    # DEF generator()
    # - data generator module
    #-------------------------------
    def generator(self, _samples, _batch_sz):
#        print("Data Augmented?: ", self.augmented)

        num_samples = len(_samples)
        while 1:
            #shuffle(samples)
            for offset in range(0, num_samples, _batch_sz):
                batch_samples = _samples[offset:offset+_batch_sz]

                images = []
                angles = []

                for batch_sample in batch_samples:
                    fname = '/'.join([self.data_path, self.subset, "IMG", batch_sample[0].split('/')[-1]])
                    center_image = cv2.imread(fname)
                    center_angle = float(batch_sample[3])   # measurement

                    images.append(center_image)
                    angles.append(center_angle)

                    # augment data
                    if self.augmented:
                        #images.append(cv2.flip(center_image, 1))   # ANOTHER WAY: image_flipped = np.fliplr(image)
                        images.append(np.fliplr(center_image))   # ANOTHER WAY: image_flipped = np.fliplr(image)
                        angles.append(center_angle * -1.0)


                X_train = np.array(images)
                y_train = np.array(angles)
    #            print("\nlen(X_train): %d, len(y_train): %d"%(len(X_train), len(y_train)))

                yield sklearn.utils.shuffle(X_train, y_train)



    ##------------------------------
    # DEF fitGenerator()
    # - fit train data generator
    #-------------------------------
    def fitGenerator(self):
        self.train_generator = self.generator(self.train_samples)
        self.validation_generator = self.generator(self.validation_samples)
        self.spe = int(np.ceil(float(len(self.train_samples))/float(self.batch_sz)))
        print("len(train_samples): ", len(self.train_samples))
        print("spe: ", self.spe)


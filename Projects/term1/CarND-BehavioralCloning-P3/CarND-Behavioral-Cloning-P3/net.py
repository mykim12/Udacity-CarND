import matplotlib.pyplot as plt
import h5py

from keras.models import Sequential#, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.utils import plot_model

import numpy as np


class NetType(object):
    BASIC = 'basic'
    LENET = 'LeNet'
    DNET = 'DeeperNet'



class BCNet(object):
    def __init__(self, _bcdata, _netType):

        self.bcd = _bcdata
        self.netType = _netType

        # create model
        self.model = Sequential()

        #ch, row, col = 3, 80, 320   # Trimmed image format
        self.channel, self.height, self.width = 3, 160, 320 
        self.batch_size = 32


    ##------------------------------
    # DEF buildNet()
    # - build a complet neural network
    #   by netType
    #-------------------------------
    def buildNet(self):
        # add data pre-processing layer
        self.prepLayer()

        # build rest of the network
        if self.netType == NetType.BASIC:
            self.basicNet()
        elif self.netType == NetType.LENET:
            self.LeNet()
        elif self.netType == NetType.DNET:
            self.dNet()

        # setup optimizer
        self.opt()

        # visualize model
        self.plotModel()



    ##------------------------------
    # DEF train()
    # - train the network
    #-------------------------------
    def train(self, _n_epoch=5):
        if self.bcd.fit_gen:
            # load data with fit_generator

            #self.bcd.fitGenerator()

            train_generator = self.bcd.generator(self.bcd.train_samples, self.batch_size)
            validation_generator = self.bcd.generator(self.bcd.validation_samples, self.batch_size)

            spe = int(np.ceil(float(len(self.bcd.train_samples))/float(self.batch_size)))
            print("len(train_samples): ", len(self.bcd.train_samples))
            print("spe: ", spe)

            self.history_object = self.model.fit_generator(train_generator, 
                                                      steps_per_epoch = spe,
                                       		      validation_data = None,
                                                      nb_val_samples = self.bcd.num_val_samples, 
                                                      nb_epoch=_n_epoch, verbose=1)
            # plot result
            self.plot()

        else:
            self.model.fit(self.bcd.X_train, self.bcd.y_train, validation_split=0.2, shuffle=True, \
                           epochs=_n_epoch, verbose=1)



    ##------------------------------
    # DEF prepLayer()
    # - data pre-processing layer
    # - mean-shift, normalize, 
    #   crop, etc
    #-------------------------------
    def prepLayer(self):
        # normalize pixel values and shift mean to be centered
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(self.height, self.width, self.channel)))
        '''
        # preprocess data, centered around zero with small std
        #model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(self.height, self.width, self.channel), 
                          output_shape=(self.height, self.width, self.channel)))
        '''
        # crop images 60 rows pixels from the top of the image, 
        #             25 rows pixels from the bottom of the image,
        #             0 columns of pixels from the left of the image,
        #         and 0 columns of pixels from the right of the image
        self.model.add(Cropping2D(cropping=((60, 25), (0, 0))))
        

    ##------------------------------
    # DEF basicNet()
    # - very basic net
    #-------------------------------
    def basicNet(self):
        self.model.add(Flatten())
        self.model.add(Dense(1))


    ##------------------------------
    # DEF LeNet()
    # - LeNet
    #   (http://yann.lecun.com/exdb/lenet/)
    #-------------------------------
    def LeNet(self):
        self.model.add(Convolution2D(6, 5, 5, activation="relu", input_shape=(75, 320, 3)))
        #self.model.add(Convolution2D(6, 5, 5, activation="relu", input_shape=(160, 320, 3)))
        self.model.add(MaxPooling2D())
        self.model.add(Convolution2D(6, 5, 5, activation="relu"))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(120))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(84))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1))


    ##------------------------------
    # DEF dNet()
    # - a bit deeper net
    #-------------------------------
    def dNet(self):
        self.model.add(Convolution2D(32, 7, 7, activation="relu"))
        self.model.add(MaxPooling2D())
        self.model.add(Convolution2D(16, 5, 5, activation="relu"))
        self.model.add(MaxPooling2D())
        self.model.add(Convolution2D(16, 5, 5, activation="relu"))
        self.model.add(AveragePooling2D())
        self.model.add(Convolution2D(16, 3, 3, activation="relu"))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))


    ##------------------------------
    # DEF opt()
    # - set net loss and optimizer
    #-------------------------------
    def opt(self):
        self.model.compile(loss='mse', optimizer='adam')



    ##------------------------------
    # DEF saveModel()
    # - save trained prarms to file
    #-------------------------------
    def saveModel(self, _mf='model.h5'):
        print("Saving model...")
        self.model.save(_mf)
        print("Model is saved to %s!"%_mf)


    ##------------------------------
    # DEF plotModel()
    # - visualize model architecture
    #-------------------------------
    def plotModel(self):
        plot_model(self.model, to_file='model_%s.png'%self.netType)

    ##------------------------------
    # DEF plot()
    # - plot training result
    #-------------------------------
    def plot(self):
        ### print the keys contained in the history object
        print(self.history_object.history.keys())

        print("Now let's plot")
        plt.ion()
        ### plot the training and validation loss for each epoch
        plt.plot(self.history_object.history['loss'])
#        plt.plot(self.history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
        plt.savefig('TrainResult_%s.png'%self.netType)
        input("DONE?")
        plt.ioff()




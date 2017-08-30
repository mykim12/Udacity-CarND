import csv
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import sklearn

from keras.models import Sequential#, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D


DATA_PATH="/data/udacity/bcloning_data"
SUBSET = "all"
#SUBSET = "provided"
samples = []

csvfile = '/'.join([DATA_PATH, SUBSET, 'driving_log.csv'])
with open(csvfile) as fp:
    reader = csv.reader(fp)
    next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    AUGMENTED = False
    print("Data Augmented?: ", AUGMENTED)

    num_samples = len(samples)
    while 1:
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                fname = '/'.join([DATA_PATH, SUBSET, "IMG", line[0].split('/')[-1]])
                center_image = cv2.imread(fname)
                center_angle = float(batch_sample[3])

                images.append(center_image)
                angles.append(center_angle)

                # augment data
                if AUGMENTED:
                    #images.append(cv2.flip(center_image, 1))   # ANOTHER WAY: image_flipped = np.fliplr(image)
                    images.append(np.fliplr(center_image))   # ANOTHER WAY: image_flipped = np.fliplr(image)
                    angles.append(center_angle * -1.0)


            X_train = np.array(images)
            y_train = np.array(angles)
#            print("\nlen(X_train): %d, len(y_train): %d"%(len(X_train), len(y_train)))

            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320   # Trimmed image format
ch, row, col = 3, 160, 320   # Trimmed image format

# create model
model = Sequential()

# normalize pixel values and shift mean to be centered
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320,3)))

# preprocess data, centered around zero with small std
#model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))

# crop images 70 rows pixels from the top of the image, 
#             25 rows pixels from the bottom of the image,
#             0 columns of pixels from the left of the image,
#         and 0 columns of pixels from the right of the image
model.add(Cropping2D(cropping=((60, 25), (0, 0))))

if False:
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    #model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
elif True:
    model.add(Convolution2D(6, 5, 5, activation="relu", input_shape=(75, 320, 3)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')

# fit()
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=18, verbose = 1)

# fit_generator()
spe = int(np.ceil(float(len(train_samples))/32.0))
print("len(train_samples): ", len(train_samples))
print("spe: ", spe)

history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch = spe,
                                     validation_data = validation_generator,
                                     nb_val_samples = len(validation_samples), 
                                     nb_epoch=3, verbose=1)



print("Saving model...")
model.save('./model.h5')
print("Model saved!")


### print the keys contained in the history object
print(history_object.history.keys())

print("Now let's plot")
plt.ion()
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
a = input("DONE?")
plt.ioff()





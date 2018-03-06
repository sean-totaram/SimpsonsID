"""
...
CNN Image Classifier
Sean Totaram
...
"""

import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as k
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

k.set_image_dim_ordering('tf')

# Gets location of Dataset
workingDir = os.getcwd()
dataDir = workingDir + "\dataset\\training_set"
classNames = os.listdir(dataDir)

# IMG dimensions
height = 32
width = 32
chan = 3
epoch = 100
numClasses = len(classNames)

numImages = 0
for folder in classNames:
    curDir = os.listdir(dataDir + '\\' + folder)
    for img in curDir:
        numImages += 1

# Stores training data in np array with 4 dim
trainData = np.zeros((numImages, height, width, chan))
trainLabels = np.zeros((numImages, numClasses))
count = 0
i = 0
for folder in classNames:
    curDir = os.listdir(dataDir + '\\' + folder)
    for img in curDir:
        curImg = dataDir + '\\' + folder + '\\' + img
        trainData[count] = cv2.cvtColor(cv2.resize(cv2.imread(curImg),(height,width)),cv2.COLOR_BGR2RGB)
        trainLabels[count][i] = 1
        count += 1
    i += 1 

# Gets number of files in the test set
dataDir = workingDir + "\dataset\\test_set"
numImages = 0
for folder in classNames:
    curDir = os.listdir(dataDir + '\\' + folder)
    for img in curDir:
        numImages += 1
        
# Stores training data in np array with 4 dim
testData = np.zeros((numImages, height, width, chan))
testLabels = np.zeros((numImages, numClasses))
count = 0
i = 0
for folder in classNames:
    curDir = os.listdir(dataDir + '\\' + folder)
    for img in curDir:
        curImg = dataDir + '\\' + folder + '\\' + img
        testData[count] = cv2.cvtColor(cv2.resize(cv2.imread(curImg),(height,width)),cv2.COLOR_BGR2RGB)
        testLabels[count][i] = 1
        count += 1
    i += 1   

# Normalize data
trainData = trainData.astype('float32') / 255.0
testData = testData.astype('float32') / 255.0

# initialize CNN model
model = Sequential()
# First Conv layer
model.add(Conv2D(100, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(height, width, chan)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# Second Conv layer
model.add(Conv2D(100, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# Third Conv layer
model.add(Conv2D(100, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# Flatten Layers
model.add(Flatten())
# Output
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(numClasses, activation='softmax'))

# Complie
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fits and trains
hist = model.fit(trainData, trainLabels, batch_size = 32, nb_epoch = epoch, validation_data = (testData, testLabels))

# plot Model
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

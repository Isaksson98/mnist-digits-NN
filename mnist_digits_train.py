from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Sequential, utils
from tensorflow.keras.datasets import mnist

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize
x_train = x_train / 255.0
x_train = np.expand_dims(x_train, axis=3)

x_test = x_test / 255.0
x_test = np.expand_dims(x_test, axis=3)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Define hyperparameters
FILTER_SIZE = 3
INPUT_SIZE  = 28
MAXPOOL_SIZE = 2
BATCH_SIZE = 32
STEPS_PER_EPOCH = len(x_train)//BATCH_SIZE
EPOCHS = 15

def create_model():
    model = Sequential()

    model.add(Conv2D(32, FILTER_SIZE, activation='relu', input_shape=(INPUT_SIZE,INPUT_SIZE, 1)))
    model.add(Conv2D(32, FILTER_SIZE, activation='relu'))
    #model.add(Conv2D(32, FILTER_SIZE, activation='relu'))
    model.add(MaxPooling2D(MAXPOOL_SIZE, MAXPOOL_SIZE))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, FILTER_SIZE, activation='relu'))
    model.add(Conv2D(64, FILTER_SIZE, activation='relu'))
    #model.add(Conv2D(64, FILTER_SIZE, activation='relu'))
    model.add(MaxPooling2D(MAXPOOL_SIZE, MAXPOOL_SIZE))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()

aug = ImageDataGenerator(rotation_range=25, zoom_range=0.15,
	width_shift_range=0.25, height_shift_range=0.25, shear_range=0.1,
	fill_mode="nearest")

model.fit(aug.flow(x_train, y_train, batch_size=BATCH_SIZE), validation_data=(x_test, y_test), steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose=2)

score = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy: {}".format(score[1]))

predictions = model.predict(x_test)

model.save('./mnistModel.h5')

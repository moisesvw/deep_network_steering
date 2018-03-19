import csv
import cv2
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

from sklearn.model_selection import train_test_split
import random
import sklearn
import pandas as pd

def generator(samples, batch_size=128, add_flip_image=False):
    """
    Generates random batches of data base on image path samples
    It could add flipped images if incoming samples contains
    Left and Rigth cameras images

    Arguments:
    samples -- list containing images paths
    batch_size -- size of batch
    add_flip_image -- whether yes or not flip images

    Returns:
    [X_train, y_train] batch for training
    """

    num_samples = len(samples)
    while True:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                data_dir = batch_sample[3].strip()
                image_path = data_dir + batch_sample[0].strip()
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                steering = float(batch_sample[2])

                if batch_sample[1] == 'L':
                    steering+= 0.25
                elif batch_sample[1] == 'R':
                    steering-= 0.25
 
                images.append(image)
                angles.append(steering)

                #Augmentation flip images randomly
                if add_flip_image and random.random() > 0.5:
                    images.append(cv2.flip(image, 1))
                    angles.append(steering * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
 
            yield sklearn.utils.shuffle(X_train, y_train)

def normalize_pixels(x):
    """
    Normalize image data
    """
    return x / 255.0 - 0.5

def train_genarator_model(model, train_generator, steps_per_epoch,
                          validation_generator, validation_steps, epochs=1):
    """
    Train a model with data generators
    Arguments:
    model -- Keras model
    train_generator -- data generator for train set
    steps_per_epoch -- how many steps for training set depends on batch size
    validation_generator -- data generator for validation set
    validation_steps -- how many steps for for validation depends on batch size
    """

    optimizer = optimizers.Adam(lr=0.0006)

    model.compile(loss='mse', optimizer=optimizer)
    model.fit_generator(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        epochs=epochs)
    return model
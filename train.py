"""This is an implementation of end to end driving steering network
   it trains the network from raw images collected from Udacity
   Self Driving Car Simulator.
"""

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

class Train:
    def __init__(self, log_path, data_path, model_name, bath_size=64,
                 epochs=1, flip_images=False):
        self.log_path = log_path
        self.data_path = data_path
        self.bath_size = bath_size
        self.epochs = epochs
        self.flip_images = flip_images
        self.model_name = model_name
        self.model = None
        self.model_evaluation = None

    def generator(self, samples, batch_size=128, add_flip_image=False):
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

    def normalize_pixels(self, x):
        """
        Normalize image data
        """
        return x / 255.0 - 0.5

    def train_genarator_model(self, model, train_generator, steps_per_epoch,
                            validation_generator, validation_steps, epochs=1):
        """
        Train a model with data generators
        Arguments:
        model -- Keras model
        train_generator -- data generator for train set
        steps_per_epoch -- how many steps for training set depends on batch size
        validation_generator -- data generator for validation set
        validation_steps -- how many steps for for validation depends on batch size

        Returns:
        a keras trained model
        """

        optimizer = optimizers.Adam(lr=0.0006)

        model.compile(loss='mse', optimizer=optimizer)
        model.fit_generator(train_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=validation_generator,
                            validation_steps=validation_steps,
                            epochs=epochs)
        return model

    def get_data(self, all_cameras=False, data_source={"./tmp/driving_log.csv": "./tmp/images/"}):
        """
        generates image data base on its data recorded from cameras in simulator
        Arguments:
        all_cameras -- whether yes or not should use all of three car cameras or just the center one
        data_source -- dict contining data
                        key: path of csv file containing the images path and steering angle
                        value: directory where the steering images are stored
        Returns:
        list of data needed to proceed with training
        """
        samples = []
        CENTER, LEFT, RIGTH, STEERING = 0, 1, 2, 3
        for path, directory in data_source.items():
            with open(path) as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)
                for line in reader:
                    # add Left and Right cameras randomly
                    if all_cameras and random.random() > 0.7:
                        samples.append([line[CENTER], 'C',line[STEERING], directory ])
                        samples.append([line[LEFT],   'L',line[STEERING], directory ])
                        samples.append([line[RIGTH],  'R',line[STEERING], directory ])
                    else:
                        samples.append([line[CENTER], 'C',line[STEERING], directory ])

        return samples

    def nvida_model(self):
        """
        Implementation of NVIDIA Model base on paper end to end drive steering
        This model predicts the angle steering given an image from center camera of the car
        https://arxiv.org/abs/1604.07316

        Returns:
        model architecture
        """
        model = Sequential()
        model.add(Cropping2D(cropping=((65, 25), (0,0)), input_shape=(160,320,3)))
        model.add(Lambda(self.normalize_pixels))
        model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
        model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
        model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1164))
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        return model

    def perform_training(self):
        data_source = {
            self.log_path: self.data_path
        }

        all_images_data = self.get_data(all_cameras=True, data_source=data_source)
        train_samples, validation_samples = train_test_split(all_images_data, test_size=0.3)
        validation_samples, test_samples  = train_test_split(validation_samples, test_size=0.5)

        generator_batch = self.bath_size
        add_flip_image = self.flip_images

        train_generator = self.generator(train_samples, batch_size=generator_batch, add_flip_image=add_flip_image)
        validation_generator = self.generator(validation_samples, batch_size=generator_batch)
        test_generator = self.generator(test_samples, batch_size=generator_batch)

        train_steps      =  ((len(train_samples)*2)/generator_batch) if add_flip_image else len(train_samples)/generator_batch
        validation_steps = len(validation_samples)/generator_batch

        model = self.train_genarator_model(self.nvida_model(),
                                    train_generator, train_steps,
                                    validation_generator, validation_steps,
                                    epochs=self.epochs)

        nvda_eval = model.evaluate_generator(test_generator, len(test_samples)/generator_batch)

        self.model = model
        self.model_evaluation = nvda_eval 
        model.save(self.model_name)
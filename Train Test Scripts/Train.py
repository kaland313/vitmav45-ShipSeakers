########################################################################################################################
# Imports
########################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import imageio
import cv2

from keras.applications import imagenet_utils
from keras.utils import Sequence
from keras import backend as K
from keras.utils import plot_model

import keras.models as models
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD

# import json
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# from tensorflow.python.client import device_lib


########################################################################################################################
# Import Function definitions
########################################################################################################################
from ShipSegFunctions import*

########################################################################################################################
# GPU info
########################################################################################################################
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
# Device check
# print(device_lib.list_local_devices())

print(K.tensorflow_backend._get_available_gpus())

########################################################################################################################
# PARAMETERS
########################################################################################################################
image_path = "/run/media/kalap/Storage/Deep learning 2/train_v2"
segmentation_data_file_path = '/run/media/kalap/Storage/Deep learning 2/train_ship_segmentations_v2.csv'
# image_path = "../data/train_img"
# segmentation_data_file_path = '../data/train_ship_segmentations_v2.csv'
valid_split = 0.15
test_split = 0.15

# resize_img_to = (768, 768)
# resize_img_to = (384, 384)
resize_img_to = (192, 192)
batch_size = 8

########################################################################################################################
# Load and prepare the data
########################################################################################################################

# Load the file which contains the masks for each image
df_train = pd.read_csv(segmentation_data_file_path)

# Look for missing files and remove them from the dataframe
# import os
# img_files = os.listdir('../data/train_img')
# df_train['img_found'] = False
# for img_file in img_files:
#     df_train['img_found'] = (df_train['img_found']) | (df_train['ImageId']==img_file)
# df_train = df_train[df_train['img_found']]
# df_train = df_train.drop('img_found',axis = 1)
# df_train = df_train.reset_index(drop=True)

# Split the data
train_img_ids, valid_img_ids, test_img_ids = separate(df_train.index.values, valid_split, test_split)

# Define the generators
rgb_channels_number = 3
dimension_of_the_image = resize_img_to


training_generator = DataGenerator(
    train_img_ids,
    df_train,
    image_path,
    batch_size=batch_size,
    dim=dimension_of_the_image,
    n_channels=rgb_channels_number
)

validation_generator = DataGenerator(
    valid_img_ids,
    df_train,
    image_path,
    batch_size=batch_size,
    dim=dimension_of_the_image,
    n_channels=rgb_channels_number
)

# Test the generators
# gen_img, gen_mask = training_generator.__getitem__(10)
# print(gen_img.shape, gen_mask.shape)
# disp_image_with_map(gen_img[0], gen_mask[0])

#178573
# print(df_train.at[178573, 'ImageId'],df_train.at[178573, 'EncodedPixels'])

########################################################################################################################
# Build the network
########################################################################################################################

def SegNet(input_layer):
    kernel = 3
    filter_size = 32
    pool_size = 2
    residual_connections = []

    x = Conv2D(filter_size, kernel, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    residual_connections.append(x)

    x = Conv2D(64, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    kernel = 3
    filter_size = 32
    pool_size = 2

    x = Conv2D(64, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, residual_connections[0]])

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = Conv2D(filter_size, kernel, padding='same')(x)
    x = BatchNormalization()(x)

    return x

def Unet(input_layer):
    kernel = 3
    filter_size = 32
    pool_size = 2

    x = Conv2D(filter_size, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filter_size, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(filter_size*2, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filter_size*2, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = Conv2D(filter_size*4, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filter_size*4, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = Conv2D(filter_size*2, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filter_size*2, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = Conv2D(filter_size, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filter_size, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    return x

########################################################################################################################
classes = 1

input_layer = Input((*dimension_of_the_image, 3))

decoded_layer = Unet(input_layer)

final_layer = Conv2D(classes, 1, padding='same', activation='sigmoid')(decoded_layer)

model = Model(inputs=input_layer, outputs=final_layer)

opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='binary_crossentropy')
print(model.summary())

plot_model(model, to_file='model.png', show_shapes=True)

########################################################################################################################
# Train the network
########################################################################################################################
patience=40
early_stopping=EarlyStopping(patience=patience, verbose=1)
checkpointer=ModelCheckpoint(filepath='model.hdf5', save_best_only=True, verbose=1)
# history = model.fit_generator(generator=training_generator,
#                     steps_per_epoch=len(training_generator),
#                     epochs=1000,
#                     validation_data=validation_generator,
#                     validation_steps=len(validation_generator),
#                     callbacks=[checkpointer, early_stopping],
#                     verbose=1)
history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=200,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=200,
                    callbacks=[checkpointer, early_stopping],
                    verbose=1)

plot_history(history)
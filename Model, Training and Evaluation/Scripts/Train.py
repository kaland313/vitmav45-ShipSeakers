########################################################################################################################
# Imports
########################################################################################################################
import numpy as np
import pandas as pd
from tqdm import tqdm
# import imageio
import cv2
from tqdm import tqdm

from keras.applications import imagenet_utils
from keras.utils import Sequence
from keras import backend as K
from keras.utils import plot_model

import keras.models as models
from keras.models import load_model
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, CSVLogger
from keras.optimizers import SGD, Adam

########################################################################################################################
# Import Function definitions
########################################################################################################################
from ShipSegFunctions import*

########################################################################################################################
# GPU info
########################################################################################################################
# import json
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.client import device_lib
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# Device check
# print(device_lib.list_local_devices())

# print(K.tensorflow_backend._get_available_gpus())

########################################################################################################################
# PARAMETERS
########################################################################################################################
# Data on AWS
image_path = "/data/train_v2"
segmentation_data_file_path = '/data/train_ship_segmentations_v2.csv'
# Data locally
# image_path = "/run/media/kalap/Storage/Deep learning 2/train_v2"
# segmentation_data_file_path = '/run/media/kalap/Storage/Deep learning 2/train_ship_segmentations_v2.csv'
# Data on Github
# image_path = "../data/train_img"
# segmentation_data_file_path = '../data/train_ship_segmentations_v2.csv'

valid_split = 0.15
test_split = 0.15

# resize_img_to = (768, 768)
# resize_img_to = (384, 384)
resize_img_to = (256, 256)
# resize_img_to = (192, 192)
batch_size = 16

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
# df_train = df_train.drop('img_found', axis=1)
# df_train = df_train.reset_index(drop=True)

# Drop corrupted images
# List source: https://www.kaggle.com/iafoss/unet34-dice-0-87
exclude_list = ['6384c3e78.jpg', '13703f040.jpg', '14715c06d.jpg', '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images
for imgId in exclude_list:
    df_train = df_train[~(df_train['ImageId'] == imgId)]

# Drop images without ships
df_train = df_train[~df_train['EncodedPixels'].isnull()]

print(df_train.describe())

# Split the data
train_img_ids, valid_img_ids, test_img_ids = separate(df_train['ImageId'].values, valid_split, test_split)
np.save("test_img_ids.npy", test_img_ids)
np.save("valid_img_ids.npy", valid_img_ids)
np.save("train_img_ids.npy", train_img_ids)

# Define the generators
training_generator = DataGenerator(
    train_img_ids,
    df_train,
    image_path,
    batch_size=batch_size,
    dim=resize_img_to,
    split_to_sub_img = True,
    shuffle_on_every_epoch=False,
    shuffle_on_init = True
)

validation_generator = DataGenerator(
    valid_img_ids,
    df_train,
    image_path,
    batch_size=batch_size,
    dim=resize_img_to,
    split_to_sub_img = True,
    forced_len = 25
)

# # Test the generators
# gen_img, gen_mask  = training_generator.__getitem__(0)
# gen_ids = training_generator.get_last_batch_ImageIDs()
# for iii in range(4):
#     # print(gen_img.shape, gen_mask.shape)
#     disp_image_with_map(gen_img[iii], gen_mask[iii], gen_ids[iii])
#     # training_generator.on_epoch_end()


########################################################################################################################
# Model definition
########################################################################################################################
def Unet_encoder_layer(input_layer,kernel,filter_size,pool_size):
    x = Conv2D(filter_size, kernel, padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filter_size, kernel, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    residual_connection = x
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    return x, residual_connection


def Unet_decoder_layer(input_layer,kernel,filter_size,pool_size,residual_connection):
    filter_size = int(filter_size)
    x = UpSampling2D(size=(pool_size, pool_size))(input_layer)
    x = Concatenate()([residual_connection, x])
    x = Conv2D(filter_size, kernel, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filter_size, kernel, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    return x


def Unet(input_layer):
    kernel = 3
    filter_size = 64
    pool_size = 2
    residual_connections = []

    x, residual_connection = Unet_encoder_layer(input_layer, kernel, filter_size, pool_size)
    residual_connections.append(residual_connection)

    filter_size *= 2
    x, residual_connection = Unet_encoder_layer(x, kernel, filter_size, pool_size)
    residual_connections.append(residual_connection)

    filter_size *= 2
    x, residual_connection = Unet_encoder_layer(x, kernel, filter_size, pool_size)
    residual_connections.append(residual_connection)

    # filter_size *= 2
    # x, residual_connection = Unet_encoder_layer(x, kernel, filter_size, pool_size)
    # residual_connections.append(residual_connection)

    filter_size *= 2
    x = Conv2D(filter_size, kernel, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filter_size, kernel, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # filter_size /= 2
    # x = Unet_decoder_layer(x, kernel, filter_size, pool_size, residual_connections[-1])
    # residual_connections = residual_connections[:-1]

    filter_size /= 2
    x = Unet_decoder_layer(x, kernel, filter_size, pool_size, residual_connections[-1])
    residual_connections = residual_connections[:-1]

    filter_size /= 2
    x = Unet_decoder_layer(x, kernel, filter_size, pool_size, residual_connections[-1])
    residual_connections = residual_connections[:-1]

    filter_size /= 2
    x = Unet_decoder_layer(x, kernel, filter_size, pool_size, residual_connections[-1])

    final_layer = Conv2D(1, 1, padding='same', activation='sigmoid')(x)

    return final_layer


########################################################################################################################
# Build the model
########################################################################################################################
input_layer = Input((None, None, 3))
output_layer = Unet(input_layer)

model = load_model("model.hdf5", custom_objects={'dice_coef_loss': dice_coef_loss})
# model = Model(inputs=input_layer, outputs=output_layer)

# opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss=dice_coef_loss)
print(model.summary())

f = open('Training history.txt', 'a')
# f = open('Training history.txt', 'w')
# f.write(str(device_lib.list_local_devices()))
# f.write("\n\n")
# model.summary(print_fn=lambda x: f.write(x + '\n'))
# f.write("\n\n")
f.write('epoch: ' + '\tloss: ' + '\tval_loss: ' + '\n')

plot_model(model, to_file='model.png', show_shapes=True)

########################################################################################################################
# Train the network
########################################################################################################################
early_stopping = EarlyStopping(patience=50, verbose=1)
checkpoint = ModelCheckpoint(filepath='model.hdf5', save_best_only=True, verbose=1)
csv_logger = CSVLogger('Training log.csv', separator=';',append=True)
logger = LambdaCallback(on_epoch_end=lambda epoch, logs: f.write(str(epoch) +'\t'
                                                                 + str(logs['loss']) +'\t'
                                                                 + str(logs['val_loss']) + '\t'
                                                                 + '\n'),
                        on_train_end=lambda logs: f.close())

history = model.fit_generator(generator=training_generator,
                              steps_per_epoch=250,
                              epochs=500,
                              validation_data=validation_generator,
                              validation_steps=len(validation_generator),
                              callbacks=[checkpoint, logger, csv_logger],
                              verbose=1)


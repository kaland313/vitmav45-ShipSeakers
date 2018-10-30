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

# import json
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# from tensorflow.python.client import device_lib

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
# Function definitions
########################################################################################################################
cmap = pl.cm.viridis
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)

def preprocess_input(x):
    """Preprocesses a Numpy array encoding a batch of images.
    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
        - tf: will scale pixels between -1 and 1, sample-wise.
        """
    return imagenet_utils.preprocess_input(x, mode='tf')


def read_transform_image(img_file_path):
    """
    Load the image specified by img_file_path from the filesystem.
    If we will aply transformations on the images (for data augmentation), this function will be responsible for that

    Retunrns the image in a (768, 768, 3) array
    """
    img = imageio.imread(img_file_path)
    # print(img)
    return preprocess_input(img)


def separate(train_data, valid_split=0.2, train_split=0.2):
    """
    Separate the dataset into 3 different part. Train, validation and test.
    train_data and test_data sets are 1D numpy arrays.

    returns the train, valid and test data sets
    """

    sum_ = train_data.shape[0]
    train = None
    valid = None
    test = None

    train = train_data[:int(sum_ * (1 - valid_split - test_split))]
    valid = train_data[int(sum_ * (1 - valid_split - test_split)):int(sum_ * (1 - test_split))]
    test = train_data[int(sum_ * (1 - test_split)):]

    return train, valid, test


def rle_decode(mask_rle, shape=(768, 768)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - ship 0 - background
    If mask_rle mask is nan, the returned numpy array only contains zeros
    """
    # Create the all zero mask
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    if mask_rle == mask_rle:  # if mask_rle is nan that this equality check returns false and the mask array remains 0
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths

        for lo, hi in zip(starts, ends):
            mask[lo:hi] = 1

    return mask.reshape(shape).T  # Needed to align to RLE direction

def disp_image_with_map(img_matrix, mask_matrix):
    """
    Displays the image image with the mask layed on top of it. Yellow highlight indicates the ships.
    my_cmap is a color map which is transparent at one end it.
    """
    plt.imshow(img_matrix*0.5+0.5)
    plt.imshow(mask_matrix[:,:,0], alpha=0.5, cmap=my_cmap)
    plt.axis('off')
    plt.show()


# Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):

    def __init__(self, list_IDs, ship_seg_df, img_path_prefix, batch_size=32, dim=(32, 32, 32),
                 n_channels=1, n_classes=10, shuffle=True):
        # Initialization
        self.dim = dim  # dataset's dimension
        self.img_prefix = img_path_prefix  # location of the dataset
        self.batch_size = batch_size  # number of data/epoch
        self.ship_seg_df = ship_seg_df  # a dataframe storing the filenames and ship masks
        self.list_IDs = list_IDs  # indexes for the given subset referring to the rows of ship_seg_df
        self.n_channels = n_channels  # number of rgb chanels
        # self.n_classes = n_classes                 # ???
        self.shuffle = shuffle  # shuffle the data

        self.on_epoch_end()

    def on_epoch_end(self):
        # Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        # print("Ep end: {0}".format(self.indexes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # print("getitem: {}".format(self.indexes))

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # print("getitem: {}".format(list_IDs_temp))

        # Generate data
        X, Y = self.generate(list_IDs_temp)

        return X, Y

    def generate(self, tmp_list):
        # Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, 1))
        # print(X.shape)
        # print(X)

        # Generate data
        for i, ID in enumerate(tmp_list):
            # read image and mask referred by "ID"
            mask = rle_decode(self.ship_seg_df.at[ID, 'EncodedPixels'])
            if self.dim == (768, 768):
                X[i] = read_transform_image(self.img_prefix + "/" + self.ship_seg_df.at[ID, 'ImageId'])
                Y[i] = np.atleast_3d(mask)
            else:
                X[i] = cv2.resize(read_transform_image(self.img_prefix + "/" + self.ship_seg_df.at[ID, 'ImageId']), *self.dim)
                Y[i] = np.atleast_3d(cv2.resize(mask, *self.dim, interpolation=cv2.INTER_NEAREST))

        return X, Y


########################################################################################################################
# PARAMETERS
########################################################################################################################
image_path = "../data/train_img"
segmentation_data_file_path = '../data/train_ship_segmentations_v2.csv'
valid_split = 0.15
test_split = 0.15

resize_img_to = (768, 768) #(196,196)
batch_size = 1

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


########################################################################################################################
# Build the network
########################################################################################################################

def create_encoding_layers_residual(input_layer):
    kernel = 3
    filter_size = 64
    pool_size = 2
    residual_connections = []

    x = Conv2D(filter_size, kernel, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    residual_connections.append(x)

    x = Conv2D(128, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    residual_connections.append(x)

    x = Conv2D(256, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    residual_connections.append(x)

    x = Conv2D(512, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x, residual_connections


def create_decoding_layers_residual(input_layer, residual_connections):
    kernel = 3
    filter_size = 64
    pool_size = 2

    x = Conv2D(512, kernel, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Concatenate()([x, residual_connections[2]])

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = Conv2D(256, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, residual_connections[1]])

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = Conv2D(128, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, residual_connections[0]])

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = Conv2D(filter_size, kernel, padding='same')(x)
    x = BatchNormalization()(x)

    return x

########################################################################################################################
classes = 1

input_layer = Input((*dimension_of_the_image, 3))

encoded_layer, residual_conns = create_encoding_layers_residual(input_layer)
decoded_layer = create_decoding_layers_residual(encoded_layer, residual_conns)

final_layer = Conv2D(classes, 1, padding='same')(decoded_layer)
final_layer = Activation('softmax')(final_layer)

model = Model(inputs=input_layer, outputs=final_layer)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

plot_model(model, to_file='model.png', show_shapes=True)

########################################################################################################################
# Train the network
########################################################################################################################
patience=40
early_stopping=EarlyStopping(patience=patience, verbose=1)
checkpointer=ModelCheckpoint(filepath='fully_connected_weights.hdf5', save_best_only=True, verbose=1)
model.fit_generator(generator=training_generator,
                    steps_per_epoch=len(training_generator),
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator),
                    callbacks=[checkpointer, early_stopping],
                    verbose=1)


########################################################################################################################
# Test the network
########################################################################################################################
# test_generator = DataGenerator(
#     test_img_ids,
#     df_train,
#     image_path,
#     batch_size=batch_size,
#     dim=dimension_of_the_image,
#     n_channels=rgb_channels_number
# )

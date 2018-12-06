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

from keras.models import load_model
from keras import backend as K

import keras.models as models
from keras.models import Model

########################################################################################################################
# Import Function definitions
########################################################################################################################
from ShipSegFunctions import *

########################################################################################################################
# GPU info
########################################################################################################################import tensorflow as tf
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(device_count = {'GPU': 0}) # Use CPU for the testing
# config = tf.ConfigProto() # Use GPU
# config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# Device check
# print(device_lib.list_local_devices())
# print(K.tensorflow_backend._get_available_gpus())

########################################################################################################################
# PARAMETERS
########################################################################################################################
image_path = "/run/media/kalap/Storage/Deep learning 2/train_v2"
segmentation_data_file_path = '/run/media/kalap/Storage/Deep learning 2/train_ship_segmentations_v2.csv'
# model_path = "../Train Test Scripts on AWS/Scripts/"
model_path = ""

# resize_img_to = (768, 768)
resize_img_to = (192, 192)
batch_size = 16

########################################################################################################################
# Load and prepare the data
########################################################################################################################

# Load the file which contains the masks for each image
df_train = pd.read_csv(segmentation_data_file_path)

# Load the test data ids saved by the Train file
test_img_ids = np.load(model_path + "test_img_ids.npy")

train_img_ids = np.load(model_path + "train_img_ids.npy")


#######################################################################################################################
# Load the network
########################################################################################################################
model = load_model(model_path + "model.hdf5", custom_objects={'dice_coef_loss': dice_coef_loss})
print(model.summary())
########################################################################################################################
# Test the network
########################################################################################################################
print("#################################################################")
print("# Testing")
print("#################################################################")

test_generator = DataGenerator(
    test_img_ids,
    df_train,
    image_path,
    batch_size=batch_size,
    dim=resize_img_to,
    shuffle_on_every_epoch=False,
    shuffle_on_init=False,
    split_to_sub_img=False
)


# predictions = model.predict_generator(test_generator,
#                                       steps=1, verbose =1)


for i in range(10):
    test_images, test_mask_true = test_generator.__getitem__(i)
    test_ids, unique_ids = np.unique(test_generator.get_last_batch_ImageIDs(), return_index=True, axis=0)
    test_images = test_images[unique_ids]
    test_mask_true = test_mask_true[unique_ids]

    predictions = model.predict(test_images, verbose=1)
    disp_image_with_map2(test_images[0], test_mask_true[0], predictions[0])

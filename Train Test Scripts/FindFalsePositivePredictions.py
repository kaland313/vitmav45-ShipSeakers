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
########################################################################################################################
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(device_count = {'GPU': 0}) # Use CPU for the testing
# config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# Device check
# print(device_lib.list_local_devices())
print(K.tensorflow_backend._get_available_gpus())

########################################################################################################################
# PARAMETERS
########################################################################################################################
image_path = "/run/media/kalap/Storage/Deep learning 2/train_v2"
segmentation_data_file_path = '/run/media/kalap/Storage/Deep learning 2/train_ship_segmentations_v2.csv'
model_path = "../Train Test Scripts on AWS/Scripts/"
# model_path = ""

# resize_img_to = (768, 768)
# resize_img_to = (384, 384)
resize_img_to = (256, 256)
# resize_img_to = (192, 192)
batch_size = 2

########################################################################################################################
# Load and prepare the data
########################################################################################################################

# Load the file which contains the masks for each image
df_train = pd.read_csv(segmentation_data_file_path)

# Drop corrupted images
# List source: https://www.kaggle.com/iafoss/unet34-dice-0-87
exclude_list = ['6384c3e78.jpg', '13703f040.jpg', '14715c06d.jpg', '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images
for imgId in exclude_list:
    df_train = df_train[~(df_train['ImageId'] == imgId)]

# Keep only images without ships
df_train_dropped = df_train[df_train['EncodedPixels'].isnull()]

# Drop all images which are in the validation or the test partition
test_img_ids = np.load(model_path + "test_img_ids.npy")
valid_img_ids = np.load(model_path + "valid_img_ids.npy")
for imgId in tqdm(test_img_ids):
    df_train = df_train[~(df_train['ImageId'] == imgId)]

for imgId in tqdm(valid_img_ids):
    df_train = df_train[~(df_train['ImageId'] == imgId)]

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


training_generator = DataGenerator(
    df_train_dropped['ImageId'].values,
    df_train_dropped,
    image_path,
    batch_size=batch_size,
    dim=resize_img_to,
    split_to_sub_img=False,
    shuffle_on_every_epoch=False
)


# predictions = model.predict_generator(test_generator,
#                                       steps=1, verbose =1)

false_positive_img_ids = []

for i in tqdm(range(len(training_generator))):
    test_images, test_mask_true = training_generator.__getitem__(i)
    test_img_id = training_generator.get_last_batch_ImageIDs()
    predictions = model.predict(test_images, verbose=0)
    for iii in range(batch_size):
        # Only images without ships are presented to the network -> if any pixels are predicted as ship, it is a false positive
        if K.sum(K.flatten(predictions[iii])) > 0:
            false_positive_img_ids.append(test_img_id)
            disp_image_with_map2(test_images[iii], test_mask_true[iii], predictions[iii])
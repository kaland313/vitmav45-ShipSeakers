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
print(K.tensorflow_backend._get_available_gpus())

########################################################################################################################
# PARAMETERS
########################################################################################################################
image_path = "/run/media/kalap/Storage/Deep learning 2/train_v2"
segmentation_data_file_path = '/run/media/kalap/Storage/Deep learning 2/train_ship_segmentations_v2.csv'

resize_img_to = (192, 192)
batch_size = 8

########################################################################################################################
# Load and prepare the data
########################################################################################################################

# Load the file which contains the masks for each image
df_train = pd.read_csv(segmentation_data_file_path)

# Load the test data ids saved by the Train file
test_img_ids = np.load("test_img_ids.npy")


########################################################################################################################
# Load the network
########################################################################################################################
model = load_model("model.hdf5")

########################################################################################################################
# Test the network
########################################################################################################################
print("#################################################################")
print("# Testing")
print("#################################################################")

rgb_channels_number = 3

test_generator = DataGenerator(
    test_img_ids,
    df_train,
    image_path,
    batch_size=batch_size,
    dim=resize_img_to,
    n_channels=rgb_channels_number
)


# predictions = model.predict_generator(test_generator,
#                                       steps=1, verbose =1)

test_images, test_mask_true = test_generator.__getitem__(10)

predictions = model.predict(test_images,
                            verbose=1)
for i in range(5):
    disp_image_with_map2(test_images[i], test_mask_true[i], predictions[i])
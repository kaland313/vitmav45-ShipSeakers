########################################################################################################################
# Imports
########################################################################################################################
import numpy as np
import pandas as pd
import imageio
import cv2
from tqdm import tqdm

from keras.applications import imagenet_utils
from keras.utils import Sequence
from keras import backend as K

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


def separate(data, valid_split=0.2, test_split=0.):
    """
    Separate the dataset into 3 different part. Train, validation and test.
    train_data and test_data sets are 1D numpy arrays.

    returns the train, valid and test data sets
    """

    sum_ = data.shape[0]

    train = data[:int(sum_ * (1 - valid_split - test_split))]
    valid = data[int(sum_ * (1 - valid_split - test_split)):int(sum_ * (1 - test_split))]
    test = data[int(sum_ * (1 - test_split)):]

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

# Reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):

    def __init__(self, list_IDs, ship_seg_df, img_path_prefix, batch_size=32, dim=(32, 32), split_to_sub_img=True,
                 n_channels=3, shuffle_on_every_epoch=True, shuffle_on_init = True, forced_len=0):
        # Initialization
        self.dim = dim  # dataset's dimension
        self.img_prefix = img_path_prefix  # location of the dataset
        self.batch_size = batch_size  # number of data/epoch
        self.ship_seg_df = ship_seg_df.copy()  # a dataframe storing the filenames and ship masks
        self.list_IDs = list_IDs  # a list containing image names to be used by the generator
        self.n_channels = n_channels  # number of rgb chanels

        self.forced_len = forced_len #Needed due to a bug in predict_generator

        # defines how many sub images should be generated. 50% overlap is used, so this should be an odd number
        # self.sub_img_ratio = split_img_ratio

        self.split_to_sub_img = split_to_sub_img
        #  Due to the 50% overlap of the number of sub images per whole image is:
        self.sub_img_count = int((768.0/dim[0]))**2 + int(((768.0/dim[0])-1))**2
        self.sub_img_idx = 0
        self.sub_img_loc = [0, 0]

        # shuffle the data on after each epoch so data is split into different batches in every epoch
        self.shuffle_on_every_epoch = shuffle_on_every_epoch
        if shuffle_on_init:
            self.shuffle_data()
        else:
            self.indexes = np.arange(len(self.list_IDs))

        self.list_IDs_temp = []

    def on_epoch_end(self):
        if self.shuffle_on_every_epoch:
            self.shuffle_data()

        self.sub_img_idx = (self.sub_img_idx + 1) % self.sub_img_count

        self.sub_img_loc[0] = self.sub_img_loc[0] + self.dim[0]
        if self.sub_img_loc[0] + self.dim[0] > 768:
            # new row
            if self.sub_img_idx >= int((768.0 / self.dim[0]))**2 - 1:
                self.sub_img_loc[0] = int(self.dim[0] * 0.5)
            else:
                self.sub_img_loc[0] = 0

            self.sub_img_loc[1] = self.sub_img_loc[1] + self.dim[1]

        if self.sub_img_loc[1] + self.dim[1] > 768:
            if self.sub_img_idx >= int((768.0 / self.dim[0])) ** 2 - 1:
                self.sub_img_loc[0] = int(self.dim[0] * 0.5)
                self.sub_img_loc[1] = int(self.dim[1] * 0.5)
            else:
                # restart from the first corner
                self.sub_img_loc = [0, 0]

        # print(self.sub_img_loc, self.sub_img_idx)

    def shuffle_data(self):
        # Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)

    def __len__(self):
        # Denotes the number of batches per epoch'
        if self.forced_len == 0:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return self.forced_len

    def __getitem__(self, index):
        # Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.generate(self.list_IDs_temp)
        return X, Y

    def get_last_batch_ImageIDs(self):
        return self.list_IDs_temp

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
            mask_list = self.ship_seg_df['EncodedPixels'][self.ship_seg_df['ImageId'] == ID].tolist()
            mask = np.zeros((768, 768))
            for mask_coded in mask_list:
                mask += rle_decode(mask_coded)

            img = read_transform_image(self.img_prefix + "/" + ID)
            
            # Get sub image or resize
            if self.dim == (768, 768):
                X[i] = img
                Y[i] = np.atleast_3d(mask)
            else:
                if self.split_to_sub_img:
                    X[i] = img[self.sub_img_loc[0]: self.sub_img_loc[0]+self.dim[0],
                               self.sub_img_loc[1]: self.sub_img_loc[1] + self.dim[1]]
                    Y[i] = np.atleast_3d(mask[self.sub_img_loc[0]: self.sub_img_loc[0] + self.dim[0],
                                              self.sub_img_loc[1]: self.sub_img_loc[1] + self.dim[1]])
                else:
                    X[i] = cv2.resize(img, self.dim)
                    Y[i] = np.atleast_3d(cv2.resize(mask, self.dim, interpolation=cv2.INTER_NEAREST))

        return X, Y


########################################################################################################################
# Define custom losses
########################################################################################################################
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

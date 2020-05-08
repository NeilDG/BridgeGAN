# -*- coding: utf-8 -*-


DATASET_BIRD_NORMAL_PATH = "E:/GTA Bird Dataset/crop_img/"
DATASET_BIRD_HOMOG_PATH = "E:/GTA Bird Dataset/homo_img/"
DATASET_BIRD_GROUND_TRUTH_PATH = "E:/GTA Bird Dataset/bird_gt/"

DATASET_VEMON_FRONT_PATH = "E:/VEMON Dataset/frames/"
BIRD_IMAGE_SIZE = (320, 192) #320 x 192 original
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
NORMAL_IMAGE_SIZE = 128
TOPDOWN_IMAGE_SIZE = 128
FIG_SIZE = NORMAL_IMAGE_SIZE / 4

SAVE_FIG_PATH = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/"
TENSORBOARD_PATH = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-GenerativeExperiment/train_plot/"
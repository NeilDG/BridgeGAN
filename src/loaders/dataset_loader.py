# -*- coding: utf-8 -*-
"""
Dataset loader
Created on Fri Jun  7 19:01:36 2019

@author: delgallegon
"""

import torch
from torch.utils import data
from loaders import image_dataset
import constants
import os
from torchvision import transforms


def assemble_train_data_old(num_image_to_load = -1):
    normal_list = []; topdown_list = []
    
    images = os.listdir(constants.DATASET_PATH_NORMAL)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
    
    for i in range(image_len): #len(images)
        normal_img_path = constants.DATASET_PATH_NORMAL + images[i]
        topdown_img_path = constants.DATASET_PATH_TOPDOWN +  images[i].replace("grdView", "satView_polish")
        #print(normal_img_path + "  "  + topdown_img_path)
        normal_list.append(normal_img_path)
        topdown_list.append(topdown_img_path)
        
    return normal_list, topdown_list

def assemble_train_data(num_image_to_load = -1):
    normal_list = []; topdown_list = []; homog_list = []
    
    #load normal images
    images = os.listdir(constants.DATASET_BIRD_NORMAL_PATH)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
        
    for i in range(image_len):
        img_path = constants.DATASET_BIRD_NORMAL_PATH + images[i]
        normal_list.append(img_path)
        
    #load homog images
    images = os.listdir(constants.DATASET_BIRD_HOMOG_PATH)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
    
    for i in range(image_len):
        img_path = constants.DATASET_BIRD_HOMOG_PATH + images[i]
        homog_list.append(img_path)
    
    #load topdown images
    images = os.listdir(constants.DATASET_BIRD_GROUND_TRUTH_PATH)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
    
    for i in range(image_len):
        img_path = constants.DATASET_BIRD_GROUND_TRUTH_PATH + images[i]
        topdown_list.append(img_path)
    return normal_list, homog_list, topdown_list

def assemble_test_data(num_image_to_load = -1):
    normal_list = []
    
    #load normal images
    images = os.listdir(constants.DATASET_VEMON_FRONT_PATH)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
        
    for i in range(image_len):
        img_path = constants.DATASET_VEMON_FRONT_PATH + images[i]
        normal_list.append(img_path)

    return normal_list
def load_dataset(batch_size = 8, num_image_to_load = -1):
    normal_list, homog_list, topdown_list = assemble_train_data(num_image_to_load = num_image_to_load)
    print("Length of train images: ", len(normal_list), len(homog_list), len(topdown_list))

    train_dataset = image_dataset.TorchImageDataset(normal_list, homog_list, topdown_list)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=12,
        shuffle=True
    )
    
    return train_loader

def load_vemon_dataset(batch_size = 8, num_image_to_load = -1):
    normal_list = assemble_test_data(num_image_to_load)
    print("Length of test images: ", len(normal_list))
    
    
    test_dataset = image_dataset.VemonImageDataset(normal_list)
    train_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=12,
        shuffle=True
    )
    
    return train_loader
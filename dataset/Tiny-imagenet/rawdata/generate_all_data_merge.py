import io
import pandas as pd
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir
import shutil
import random
import numpy as np
import torch

target_folder = 'tiny-imagenet-200/cat/'
source_folder_train = 'tiny-imagenet-200/train/'
source_folder_val = 'tiny-imagenet-200/val/'

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

set_seed(666)


def copy_train_data(source_folder_train, target_folder):
    folder_list = os.listdir(source_folder_train)
    print("folder_num", len(folder_list))
    for folder in folder_list:
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
        train_folder = os.path.join(source_folder_train, folder) + "/images"
        train_file_list = os.listdir(train_folder)
        for train_file_name in train_file_list:
            src_file = os.path.join(train_folder,train_file_name)
            dst_folder = os.path.join(target_folder,folder)
            shutil.copy(src_file, dst_folder)
    print("finish!", source_folder_train)


def copy_val_data(source_folder_train, target_folder):
    folder_list = os.listdir(source_folder_train)
    print("folder_num", len(folder_list))
    for folder in folder_list:
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
        train_folder = os.path.join(source_folder_train, folder)
        train_file_list = os.listdir(train_folder)
        for train_file_name in train_file_list:
            src_file = os.path.join(train_folder,train_file_name)
            dst_folder = os.path.join(target_folder,folder)
            shutil.copy(src_file, dst_folder)
    print("finish!", source_folder_train)

copy_train_data(source_folder_train, target_folder)
copy_val_data(source_folder_val, target_folder)


print('done copy the all images')
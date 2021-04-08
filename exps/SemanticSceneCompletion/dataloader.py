from collections import defaultdict
import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
import sys
sys.path.append('../../furnace/')
from utils.img_utils import  normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape

class TrainPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img):
        img = normalize(img, self.img_mean, self.img_std)

        p_img = img.transpose(2, 0, 1)
        extra_dict = {}

        return p_img, extra_dict
class ValPre(object):
    def __call__(self, img):
        extra_dict = {}
        return img, extra_dict


def get_train_loader(engine, dataset, s3client):
#    '''
    data_setting = {'i_root': config.i_root_folder,
                    'g_root': config.g_root_folder,
                    'h_root':config.h_root_folder,
                    'ROOT_DIR': config.dataset_path,
                    'LOG_DIR': config.log_dir,
                    'm_root': config.m_root_folder,
                    't_source': config.t_source,
                    'e_source': config.e_source}

    train_preprocess = TrainPre(config.image_mean, config.image_std)

    train_dataset = dataset(data_setting, "train",
                            train_preprocess, config.batch_size * config.niters_per_epoch, s3client=s3client)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=False,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler

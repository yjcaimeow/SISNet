#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2020/05
# @Author  : Xiaokang Chen

import numpy as np
import torch
from datasets.BaseDataset import BaseDataset
import os
import cv2

class NYUv2(BaseDataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None, s3client=None):
        super(NYUv2, self).__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self._i_path = setting['i_root']
        self._g_path = setting['g_root']
        self._h_path = setting['h_root']
        self._m_path = setting['m_root']
        self._t_source = setting['t_source']
        self._e_source = setting['e_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._t_source
        if split_name == "val":
            source = self._e_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            item = item.strip()
            item = item.split('\t')
            img_name = item[0]
            file_names.append([img_name, None])

        return file_names


    def __getitem__(self, index):

        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        item_idx = names[0]
        t_path = os.path.join(self._i_path, 'voxel_bisenet_res34_4e-3/'+names[0]+'.npz')
           
        i_path = os.path.join(self._i_path, 'R', 'NYU'+item_idx+'_colors.png')
        g_path = os.path.join(self._g_path, 'L/'+item_idx+'.npz')
        m_path = os.path.join(self._m_path, item_idx+'.npz')
        w_path = os.path.join(self._i_path, 'T/'+item_idx+'.npz')
        item_name = item_idx
        ttt, www, mmm, ggg, tsdf = self._fetch_data(t_path, w_path, m_path, g_path)

        if self._split_name is 'train':
            ggg = torch.from_numpy(np.ascontiguousarray(ggg)).long()
            tsdf = torch.from_numpy(np.ascontiguousarray(tsdf)).float()
            mmm = torch.from_numpy(np.ascontiguousarray(mmm)).long()
            www = torch.from_numpy(np.ascontiguousarray(www)).float()
            ttt = torch.from_numpy(np.ascontiguousarray(ttt)).float()

        output_dict = dict(lll=ggg, tsdf=tsdf, www=www, mmm=mmm,
                           ttt=ttt, fn=str(item_name), n=len(self._file_names))
        return output_dict

    def _fetch_data(self, t_path, w_path, m_path, g_path,dtype=None):
        www = np.load(w_path)['arr_1'].astype(np.float32).reshape(60,36,60)
        _, file_extension = os.path.splitext(t_path)	
        if file_extension in ['.npz']:
            ttt = np.load(t_path)['arr_0'].astype(np.int32)
        else:
            ttt = np.load(t_path).astype(np.int32)
        ttt = torch.LongTensor(ttt)
        ttt = self.get_one_hot( ttt, 12)
        ttt = ttt.permute(3,0,1,2)
        ggg = np.load(g_path)['arr_0'].astype(np.int64)
        tsdf = np.load(w_path)['arr_0'].astype(np.float32).reshape(1,60,36,60)
        mmm = np.load(m_path)['arr_0'].astype(np.int64)
        www = www.flatten()
        return ttt, www, mmm, ggg, tsdf

    def get_one_hot(self, label, N): ## N classes shapenet N=4, NYU N=8
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(N)
        ones = ones.index_select(0, label)
        size.append(N)
        return ones.view(*size)
    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

    @classmethod
    def get_class_names(*args):

        return ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
                'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
                'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
                'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']
    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name

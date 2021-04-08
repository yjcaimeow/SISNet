#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from config import config
import sys
#sys.path.append('/mnt/lustre/liushinan/cyj/start/furnace')
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from nyu import NYUv2
from network import Network
from dataloader import ValPre

logger = get_logger()


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        lll = data['lll']
        www = data['www']
        mmm = data['mmm']
        tsdf = data['tsdf']
        ttt = data['ttt']

        name = data['fn']
        pp, ps, pe = self.eval_ssc(ttt, tsdf, device)

        results_dict = {'pp':pp, 'll':lll, 'ww':www, 'pred': ps, 'tsdf':tsdf,
                        'name':name, 'mm':mmm}
        return results_dict

    def hist_info(self, n_cl, ppp, ggg):
        assert (ppp.shape == ppp.shape)
        k = (ggg >= 0) & (ggg < n_cl)
        labeled = np.sum(k)
        correct = np.sum((ppp[k] == ggg[k]))

        return np.bincount(n_cl * ggg[k].astype(int) + ppp[k].astype(int),
                           minlength=n_cl ** 2).reshape(n_cl,
                                                        n_cl), correct, labeled

    def compute_metric(self, results):
        hist_ssc = np.zeros((config.num_classes, config.num_classes))
        correct_ssc = 0
        labeled_ssc = 0

        tp_sc, fp_sc, fn_sc, union_sc, intersection_sc = 0, 0, 0, 0, 0
        for d in results:
            ppp = d['pp'].astype(np.int64)
            lll = d['ll'].astype(np.int64)
            www = d['ww'].astype(np.float32)
            mmm = d['mm'].astype(np.int64).reshape(-1)
            name= d['name']
            flat_ppp = np.ravel(ppp)
            flat_lll = np.ravel(lll)
            tsdf = np.ravel(d['tsdf'])

            nff = np.where((www > 0))# & (tsdf>0))
            nff_ppp = flat_ppp[nff]
            nff_lll = flat_lll[nff]

            h_ssc, c_ssc, l_ssc = self.hist_info(config.num_classes, nff_ppp, nff_lll)
            hist_ssc += h_ssc
            correct_ssc += c_ssc
            labeled_ssc += l_ssc

            ooo = (mmm == 307200) & (www > 0) & (flat_lll != 255) #& (tsdf>0)
            ooo_ppp = flat_ppp[ooo]
            ooo_lll = flat_lll[ooo]

            tp_occ = ((ooo_lll > 0) & (ooo_ppp > 0)).astype(np.int8).sum()
            fp_occ = ((ooo_lll == 0) & (ooo_ppp > 0)).astype(np.int8).sum()
            fn_occ = ((ooo_lll > 0) & (ooo_ppp == 0)).astype(np.int8).sum()

            union = ((ooo_lll > 0) | (ooo_ppp > 0)).astype(np.int8).sum()
            intersection = ((ooo_lll > 0) & (ooo_ppp > 0)).astype(np.int8).sum()

            tp_sc += tp_occ
            fp_sc += fp_occ
            fn_sc += fn_occ
            union_sc += union
            intersection_sc += intersection

        score_ssc = compute_score(hist_ssc, correct_ssc, labeled_ssc)
        IOU_sc = intersection_sc / union_sc
        precision_sc = tp_sc / (tp_sc + fp_sc)
        recall_sc = tp_sc / (tp_sc + fn_sc)
        score_sc = [IOU_sc, precision_sc, recall_sc]

        result_line = self.print_ssc_iou(score_sc, score_ssc)
        return result_line

    def eval_ssc(self, ttt, tsdf, device=None):
        sc, bsc, esc = self.val_func_process_ssc(ttt, tsdf, device)
        sc = sc.permute(1, 2, 3, 0) ## 60 36 60 12
        softmax = nn.Softmax(dim=3)
        sc = softmax(sc)
        ddddd = sc.cpu().numpy()
        pp = ddddd.argmax(3)

        return pp, ddddd, None

    def val_func_process_ssc(self, ttt, tsdf, device=None):

        tsdf = np.ascontiguousarray(tsdf[None, :], dtype=np.float32)
        tsdf = torch.FloatTensor(tsdf).cuda(device)

        ttt = np.ascontiguousarray(ttt[None, :], dtype=np.float32)
        ttt = torch.FloatTensor(ttt).cuda(device)
        with torch.cuda.device(ttt.get_device()):
            self.val_func.eval()
            self.val_func.to(ttt.get_device())
            with torch.no_grad():
                sc, ssc = self.val_func(ttt, tsdf)
                sc = sc[0]
                sc = torch.exp(sc)
        return sc, ssc, ssc

    def print_ssc_iou(self, sc, ssc):
        lines = []
        lines.append('--*-- Semantic Scene Completion --*--')
        lines.append('IOU: \n{}\n'.format(str(ssc[0].tolist())))
        lines.append('meanIOU: %f\n' % ssc[2])
        lines.append('pixel-accuracy: %f\n' % ssc[3])
        lines.append('')
        lines.append('--*-- Scene Completion --*--\n')
        lines.append('IOU: %f\n' % sc[0])
        lines.append('pixel-accuracy: %f\n' % sc[1])
        lines.append('recall: %f\n' % sc[2])

        line = "\n".join(lines)
        print(line)
        return line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = Network(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                norm_layer=nn.BatchNorm3d, eval=True)
    data_setting = {'i_root': config.i_root_folder,
                    'g_root': config.g_root_folder,
                    'h_root':config.h_root_folder,
                    'm_root': config.m_root_folder,
                    't_source': config.t_source,
                    'e_source': config.e_source}
    val_pre = ValPre()
    dataset = NYUv2(data_setting, 'val', val_pre)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)

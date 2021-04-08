from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import numpy as np
from config import config
from dataloader import get_train_loader
from network import Network
from nyu import NYUv2
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR, PolyLR
from engine.engine import Engine
from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

parser = argparse.ArgumentParser()

s3client = None

port = str(int(float(time.time())) % 20)
os.environ['MASTER_PORT'] = str(190802 + int(port))

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    train_loader, train_sampler = get_train_loader(engine, NYUv2, s3client)
    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        writer = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    if engine.distributed:
        BatchNorm2d = SyncBatchNorm
    model = Network(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                    norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in')
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    params_list = []
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,base_lr )
    optimizer = torch.optim.SGD(params_list,lr=base_lr,momentum=config.momentum,weight_decay=config.weight_decay)
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)
    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)
    engine.register_state(dataloader=train_loader, model=model,optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()
    model.train()
    print('begin train')
    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)
            minibatch = dataloader.next()
            lll = minibatch['lll']
            tsdf= minibatch['tsdf']
            www = minibatch['www']
            ttt = minibatch['ttt']
            mmm = minibatch['mmm']

            lll = lll.cuda(non_blocking=True)
            ttt = ttt.cuda(non_blocking=True)
            www = www.cuda(non_blocking=True)
            mmm = mmm.cuda(non_blocking=True)
            tsdf = tsdf.cuda(non_blocking=True)

            output, boutput = model(ttt, tsdf)
            cri_weights = 10 * torch.FloatTensor([0.010820392313388523, 0.4585814244886793, 0.0411831291920445, 0.04826918042332931, 0.33162496143513115, 0.22373353821746247, 3*0.09748478737233816, 0.1478032329336482, 0.16258443136359715, 1.0, 0.0254366993244824, 0.05126348601814224])
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none',
                                            weight=cri_weights).cuda()
            selectindex = torch.nonzero(www.view(-1)).view(-1)
            filterLabel = torch.index_select(lll.view(-1), 0, selectindex)
            filterOutput = torch.index_select(output.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex)
            loss_semantic = criterion(filterOutput, filterLabel)
            loss_semantic = torch.mean(loss_semantic)

            if engine.distributed:
                dist.all_reduce(loss_semantic, dist.ReduceOp.SUM)
                loss_semantic = loss_semantic / engine.world_size
            else:
                loss = Reduce.apply(*loss) / len(loss)

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            loss = loss_semantic
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.5f' % (loss.item())

            pbar.set_description(print_str, refresh=False)
        if engine.distributed and (engine.local_rank == 0):
            writer.add_scalar("Loss/Epoch", loss.item(), epoch)

        if (epoch > config.nepochs // 4) and ((epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1)):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
    
    if engine.distributed and (engine.local_rank == 0):
        writer.close()

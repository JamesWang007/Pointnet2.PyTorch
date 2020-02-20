import torch
import numpy as np

import _init_path
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import tensorboard_logger as tb_log
from dataset import KittiDataset
import argparse
import importlib
from pointnet2_msg import Pointnet2MSG as pointnet2_msg

import kitti_utils


def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        #log_print("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        #log_print("==> Done")
    else:
        raise FileNotFoundError

    return epoch



def getdata(pts_lidar):

    # get valid point (projected points should be in image)
    pts_rect = pts_lidar[:, 0:3]
    npoints = 16384
    
    if npoints < len(pts_rect):
        pts_depth = pts_rect[:, 2]
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        near_idxs_choice = np.random.choice(near_idxs, npoints - len(far_idxs_choice), replace=False)

        choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
            if len(far_idxs_choice) > 0 else near_idxs_choice
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, len(pts_rect), dtype=np.int32)
        if npoints > len(pts_rect):
            extra_choice = np.random.choice(choice, npoints - len(pts_rect), replace=False)
            choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)

    pts_input = pts_rect[choice, :]

    return choice, pts_input



if __name__== "__main__":
    
    FG_THRESH = 0.3
    
    # load model
    MODEL = importlib.import_module("pointnet2_msg")  # import network module
    model = MODEL.get_model(input_channels=0)
    #model = pointnet2_msg(input_channels=0)
      
    # load ckpt    
    ckpt = load_checkpoint(model, "mycode/checkpoint_epoch_100.pth")
    model.cuda()
    print("===> load checkpoint done")
    
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    pc = np.fromfile('mycode/data/train_03.bin', dtype=np.float32).reshape(-1,4)
    choice, pc_for_det = getdata(pc)
    pc_for_det = torch.from_numpy(pc_for_det.reshape(1,-1,3)).cuda(non_blocking=True).float()
        
    with torch.no_grad():
        pred_cls = model(pc_for_det)
    pred_class = (torch.sigmoid(pred_cls) > FG_THRESH).cpu().numpy()[0]  
    print("===> detection done")
    
    
    pts_rect = pc[choice, 0:3]
    res = np.concatenate((pts_rect, pred_class), axis = 1)
    
    print(res)
        
    print("glk")
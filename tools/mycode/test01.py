import torch
import numpy as np

#import os, sys
#sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

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
from 


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

if __name__== "__main__":
    
    # load model
    #MODEL = importlib.import_module("pointnet2_msg")  # import network module
    #model = MODEL.get_model(input_channels=0)
    model = 
      
    # load ckpt    
    ckpt = load_checkpoint(model, "checkpoint_epoch_100.pth")
    model.cuda()
    print("===> load checkpoint done!")
    
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    pc = np.fromfile('data/train_03.bin', dtype=np.float32).reshape(-1,4)
    pc = torch.from_numpy(pc).cuda(non_blocking=True).float()
        
    with torch.no_grad():
        pred_cls = model(pc)
    
    print(pred_cls)
        
    print("glk")
from __future__ import print_function, absolute_import
from torch.autograd import Variable
import torch.onnx
import torchvision

import argparse

import os
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
import os.path as osp
from reid import models
from reid.utils.serialization import load_checkpoint, save_checkpoint


# Create model
model = models.create('resnet50', num_features=256, dropout=0.5, num_classes=5005, cut_at_pooling=False, FCN=True)
model = nn.DataParallel(model).cuda()
dummy_input = Variable(torch.randn(32, 3, 256, 256)).cuda()

osp.join('logs/humpbackWhale/', 'checkpoint.pth.tar')


checkpoint = load_checkpoint(osp.join('logs/humpbackWhale/', 'checkpoint.pth.tar'))
model.module.load_state_dict(checkpoint['state_dict'])
torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True)

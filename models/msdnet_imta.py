from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from .msdnet_ge import msdnet_ge

__all__ = ['IMTA_MSDNet']
        
class IMTA_MSDNet(nn.Module):
    def __init__(self, args):
        super(IMTA_MSDNet, self).__init__()
        self.nBlocks = args.nBlocks
        if args.data == 'ImageNet':
            if args.step == 7:
                logits_channels = [576, 640, 608, 528, 976]
            elif args.step == 6: 
                logits_channels = [512, 544, 496, 880, 792]
            else: 
                logits_channels = [384, 384, 352, 304, 560] # step=4
        else:
            logits_channels = []
            for i in range(args.nBlocks):
                logits_channels.append(128) # 128 for cifar10/100 

        self.net = msdnet_ge(args)
        self.classifier = nn.ModuleList()
        self.isc_modules = nn.ModuleList()
        for i in range(args.nBlocks):
            if i == 0:
                in_channels = logits_channels[i]
            else:
                in_channels = logits_channels[i] * 2
            self.classifier.append(nn.Linear(in_channels, args.num_classes))
        for i in range(args.nBlocks - 1):
            out_channels = logits_channels[i + 1]
            self.isc_modules.append(nn.Sequential(
                                    nn.Linear(args.num_classes, out_channels),
                                    nn.BatchNorm1d(out_channels),
                                    nn.ReLU(inplace=True)))
    

    def forward(self, x):
        pred = []
        real_logits, logits = self.net(x)
        
        for i in range(self.nBlocks):
            if i == 0:
                in_logits = logits[i]
            else:
                in_logits = torch.cat((logits[i], feat), dim=-1)
            pd = self.classifier[i](in_logits)
            if i < self.nBlocks - 1:
                feat = self.isc_modules[i](pd)
            pred.append(pd)
        
        if self.training:
            return pred, real_logits[-1]
        else:
            return pred
            
    

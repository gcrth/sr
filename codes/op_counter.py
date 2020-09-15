from thop import clever_format
from thop import profile
from models import create_model
from data import create_dataloader, create_dataset
from utils import util
import options.options as option
import torch.multiprocessing as mp
import torch
import time
import logging
import random
import argparse
import math
import os
import faulthandler
faulthandler.enable()


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    #### create model
    model = create_model(opt)

    #### op counting
    print('Start counting')

    var_L = torch.zeros(1, 3, 320, 180).cuda()
    # var_ref=torch.zeros(1280,720).cuda()
    # var_H=torch.zeros(1280,720).cuda()
    print('netG')
    macs, params = profile(model.netG, inputs=(var_L, ))
    macs, params = clever_format([macs, params], "%.5f")
    print('macs:{},params:{}'.format(macs, params))
    # print('netD')
    # macs, params = profile(model.netD, inputs=(var_ref, ))
    # macs, params = clever_format([macs, params], "%.5f")
    # print('macs:{},params:{}'.format(macs, params))
    # print('netF')
    # macs, params = profile(model.netF, inputs=(var_H, ))
    # macs, params = clever_format([macs, params], "%.5f")
    # print('macs:{},params:{}'.format(macs, params))


if __name__ == '__main__':
    main()

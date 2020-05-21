# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import reader.task_reader as task_reader
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,5"
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样



    model = x.Model(config)
    #init_network(model)
    model=nn.DataParallel(model,device_ids=[0,1,2,3])
    model.to(config.device)
    model.load_state_dict(torch.load('THUCNews/saved_dict/ERNIE0.4gauss.ckpt'))


    train(config,model)


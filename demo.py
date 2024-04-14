from __future__ import print_function
import argparse

import torch

from utils import build_loaders, build_model, score
from __future__ import print_function
import argparse

import torch

from utils import build_loaders, build_model, score
from config import Config as C, MSRVTTLoaderConfig, MSVDLoaderConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--ckpt_fpath", type=str)
    return parser.parse_args()


def run(inputVideoPath, ckpt_fpath):
    C.loader = MSVDLoaderConfig
    checkpoint = torch.load(ckpt_fpath)
    train_iter, val_iter, test_iter, vocab = build_loaders(C)
    model = build_model(C, vocab)
    model.load_state_dict(torch.load(ckpt_fpath))
    model.cuda()
    #print(model.describe())

if __name__ == '__main__':
    args = parse_args()
    run(args.corpus, args.ckpt_fpath)





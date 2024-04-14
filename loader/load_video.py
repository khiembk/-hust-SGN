from torch.utils.data import DataLoader, Dataset, RandomSampler
import json
import os
import random
from collections import defaultdict
import gensim
import h5py
import numpy as np
import torch
def load_video_feats(fpath):
    video_feats = defaultdict(lambda: {})
    models = ['ResNet101', '3DResNext101']
    for model in models:
        fpath = "/content/-hust-SGN/" + fpath
        fin = h5py.File(fpath, 'r')
        for vid in fin.keys():
            feats = fin[vid][:]
            # Sample fixed number of frames
            sampled_idxs = np.linspace(0, len(feats) - 1, 30, dtype=int)
            feats = feats[sampled_idxs]
            video_feats[vid][model] = feats
        fin.close()
    return video_feats

def get_data_to_predict( fpath):
    data_loader = DataLoader(
            batch_size= 1,
            shuffle=False, # If sampler is specified, shuffle must be False.
            num_workers= 1,
            collate_fn= get_collate_fn(fpath))
    return data_loader

def get_collate_fn(fpath):
    pos_video_feats = load_video_feats(fpath)
    pos_vids = list(pos_video_feats.keys())
    pos_video_feats_list = defaultdict(lambda: [])

    for vid, model_feats in pos_video_feats.items():
        for model, feats in model_feats.items():
            pos_video_feats_list[model].append(feats)

    pos_video_feats_list = dict(pos_video_feats_list)

    for model in pos_video_feats_list:
        pos_video_feats_list[model] = torch.stack(pos_video_feats_list[model], dim=0).float()

    pos = (pos_vids, pos_video_feats_list)
    neg = (None,None)
    return pos, neg
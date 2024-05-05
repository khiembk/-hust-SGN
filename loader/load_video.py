import numpy as np
import h5py
from collections import defaultdict
import torch
import tqdm
# for vids, feats in tqdm(YOLO_iter, desc='score'):
#     captions = model.describe(feats)
def load_video_feats(name_feature):
    models = ['ResNet101','3DResNext101']
    video_feats = defaultdict(lambda: {})
    for model in models:
        fpath = "/content/-hust-SGN/data/MSVD/features/" +model+ name_feature
        fin = h5py.File(fpath, 'r')
        for vid in fin.keys():
            feats = fin[vid][:]
            feats_len = len(feats)
            # Sample fixed number of frames
            # lấy với khoảng cách đều
            sampled_idxs = np.linspace(0, feats_len - 1, 30, dtype=int)
            feats = feats[sampled_idxs]
            video_feats[vid][model] = feats
        fin.close()
    vids = video_feats.keys()
    return vids, video_feats
def parse_feat(video_feats):
    for model in video_feats:
        video_feats[model] = video_feats[model].cuda()
    return video_feats
#load_video_feats- > colleate_feat->prase_feat->build_iter->predict
def collate_feat(video_feats):
        video_feats_list = defaultdict(lambda: [])
        for video_feat in video_feats:
            for model, feat in video_feat.items():
                video_feats_list[model].append(feat)
        video_feats_list = dict(video_feats_list)
        for model in video_feats_list:
            video_feats_list[model] = torch.stack(video_feats_list[model], dim=0).float()
        return video_feats_list
def build_iter(vids,data_iter, batch_size= 1):
    score_dataset = {}
    for batch in iter(data_iter):
        feats = parse_feat(batch)
        for i, vid in enumerate(vids):
            feat = {}
            for model in feats:
                feat[model] = feats[model][i]
            if vid not in score_dataset:
                score_dataset[vid] = feat

    vids = score_dataset.keys()
    feats = score_dataset.values()
    while len(vids) > 0:
        vids_list = list(vids)
        vids_batch = vids_list[:batch_size]
        feats_batch = defaultdict(lambda: [])
        feats_list = list(feats)[:batch_size]
        for feat in feats_list:
            for model, f in feat.items():
                feats_batch[model].append(f)
        for model in feats_batch:
            feats_batch[model] = torch.stack(feats_batch[model], dim=0)
        yield ( vids_batch, feats_batch )
        vids_list = list(vids)
        vids = vids_list[batch_size:]
        feats_list = list(feats)
        feats = feats_list[batch_size:]

def my_test(model, data_iter, vocab):

    YOLO_iter = build_iter(data_iter)

    for  feats in tqdm(YOLO_iter):
        captions = model.describe(feats)
        print("mine captions:", captions)


import numpy as np
import h5py
from collections import defaultdict
def load_video_feats(name_feature):
    models = ['ResNet101','3DResNet101']
    video_feats = defaultdict(lambda: {})
    vids = defaultdict(lambda :{})
    for model in models:
        fpath = "/content/-hust-SGN/data/MSVD/features/" +model+ name_feature
        fin = h5py.File(fpath, 'r')
        for vid in fin.keys():
            feats = fin[vid][:]
            feats_len = len(feats)
            # Sample fixed number of frames
            sampled_idxs = np.linspace(0, feats_len - 1, 30, dtype=int)
            feats = feats[sampled_idxs]
            video_feats[vid][model] = feats
        fin.close()
    vids = list(set(vid for vid in video_feats for model in video_feats[vid]))
    return vids, video_feats
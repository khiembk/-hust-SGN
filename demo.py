from __future__ import print_function
import argparse
import torch
import utils
from utils import build_loaders, build_model, score
from config import Config as C, MSRVTTLoaderConfig, MSVDLoaderConfig
import loader.load_video

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--ckpt_fpath", type=str)
    return parser.parse_args()

def Convert (feat ):
    tensor_dict = {}
    for key, value in feat.items():
        # Convert the NumPy array to a PyTorch tensor
        tensor = torch.tensor(value, device='cuda:0')
        tensor = tensor.to(torch.float32)
        # Add the tensor to the tensor_dict
        tensor_dict[key] = tensor
    return tensor_dict
def run(inputVideoPath, ckpt_fpath):
    C.loader = MSVDLoaderConfig
    checkpoint = torch.load(ckpt_fpath)
    train_iter, val_iter, test_iter, vocab = build_loaders(C)
    model = build_model(C, vocab)
    model.load_state_dict(torch.load(ckpt_fpath))
    model.cuda()
    vids, feats=loader.load_video.load_video_feats("_test.hdf5")
    # print("Type of vids myself: ",type(vids)) #Type of vids myself:  <class 'list'>
    # print("Type of feats myself: ", type(feats)) #Type of feats myself:  <class 'collections.defaultdict'>
    # for vid in feats :
    #     print(Convert(feats[vid]))
    utils.predict(model,test_iter,vocab, vids,feats)

if __name__ == '__main__':
    args = parse_args()
    run(args.input, args.ckpt_fpath)






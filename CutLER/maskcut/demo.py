# !/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
sys.path.append('../')
import argparse
import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
from scipy import ndimage
from detectron2.utils.colormap import random_color

import dino # model
from third_party.TokenCut.unsupervised_saliency_detection import metric
from crf import densecrf
from maskcut import maskcut
from omegaconf import OmegaConf
from tqdm import tqdm
import json

import time
from numpy import random
import glob

# SelfPatch
# from SelfPatch import selfpatch_vision_transformer as vits
# from SelfPatch import utils

# import clip

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

def vis_mask(input, mask, mask_color):
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)
    

def load_dino_model(args):
    if args.pretrain_path is not None:
        url = args.pretrain_path
        
    if args.vit_arch == "small" and args.patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        feat_dim = 384
    elif args.vit_arch == "small" and args.patch_size == 8:
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        feat_dim = 384
    elif args.vit_arch == "base" and args.patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == "base" and args.patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768

    url_ = f"https://dl.fbaipublicfiles.com/dino/{url}"

    backbone = dino.ViTFeat(url_, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

    msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
    print(msg)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()
    return backbone


def load_dinov2_model(args, device_id=4):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    if not args.cpu:
        model.cuda(device_id)
    return model


def load_selfpatch_model(args, pretrained_weights="../../SelfPatch/checkpoints/dino_selfpatch.pth", arch="vit_small", patch_size=16, checkpoint_key="teacher", device_id=4):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    print(f"SelfPatch Model {arch} {patch_size}x{patch_size} built.")
    if not args.cpu: 
        model.cuda(device_id)
    utils.load_pretrained_weights(model, pretrained_weights, checkpoint_key, arch, patch_size)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def load_clip_model(args, device_id=4):
    if not args.cpu:
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    return model    


def get_prediction(args, backbone, device_id=4):
    bipartitions, _, I_new = maskcut(args.img_path, args.flow_path, backbone, args.patch_size, args.alpha, args.tau, \
        N=args.N, fixed_size=args.fixed_size, cpu=args.cpu, device_id=device_id)

    I = Image.open(args.img_path).convert('RGB')
    width, height = I.size
    pseudo_mask_list = []
    for idx, bipartition in enumerate(bipartitions):
        # post-process pesudo-masks with CRF
        pseudo_mask = densecrf(np.array(I_new), bipartition)
        pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

        # filter out the mask that have a very different pseudo-mask after the CRF
        if not args.cpu:
            mask1 = torch.from_numpy(bipartition).cuda(device_id)
            mask2 = torch.from_numpy(pseudo_mask).cuda(device_id)
        else:
            mask1 = torch.from_numpy(bipartition)
            mask2 = torch.from_numpy(pseudo_mask)
        if metric.IoU(mask1, mask2) < 0.5:
            pseudo_mask = pseudo_mask * -1

        # construct binary pseudo-masks
        pseudo_mask[pseudo_mask < 0] = 0
        pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
        pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

        pseudo_mask = pseudo_mask.astype(np.uint8)
        upper = np.max(pseudo_mask)
        lower = np.min(pseudo_mask)
        thresh = upper / 2.0
        pseudo_mask[pseudo_mask > thresh] = upper
        pseudo_mask[pseudo_mask <= thresh] = lower
        pseudo_mask_list.append(pseudo_mask)
        
    return pseudo_mask_list
    


if __name__ == "__main__":    
    args = {
        "vit_arch": "base",
        "patch_size": 16,   # 8, 14- dinov2, 16- CLIP/SelfPatch
        "vit_feat": "k",
        "alpha": 0.7,
        "tau": 0.35,
        "fixed_size": (224,224),
        "pretrain_path": None,
        "N": 1,
        "cpu": False,
        "device_id": 7
    }
    
    backbone = load_dino_model(OmegaConf.create(args))  # TODO load a backbone model.
    # backbone = load_dinov2_model(OmegaConf.create(args), device_id=args["device_id"])
    # backbone = load_stego_model(OmegaConf.create(args))
    # backbone = load_clip_model(OmegaConf.create(args), device_id=args["device_id"])
    # backbone = load_selfpatch_model(OmegaConf.create(args), device_id=args["device_id"])

    
    args["img_path"] = path/to/rgb/img  # set the image path.
    args["flow_path"] = path/to/rgb/flow # set the optical flow RGB path.
    args = OmegaConf.create(args)

    pseudo_mask_list = get_prediction(args, backbone, device_id=args.device_id)
    pseudo_mask = np.array(pseudo_mask_list[0])/255.     # normalize to [0,1]

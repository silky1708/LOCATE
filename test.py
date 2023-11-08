import logging
logging.disable(logging.CRITICAL)

import os
import torch
from pathlib import Path
from types import SimpleNamespace
from detectron2.checkpoint import DetectionCheckpointer

import config
import utils as ut
from eval_utils import eval_unsupmf, compute_metrics, list_of_dict_of_lists_to_dict_of_lists, aggregate_metrics, precision_recall, F_vanilla
from mask_former_trainer import setup, Trainer
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
from PIL import Image

import glob
import json
import torchvision.transforms as transforms
import time
# from crf import dense_crf 

torch.cuda.set_device('cuda:1')   # cuda:0

def load_model_cfg(ckpt_path, checkpoint_num='checkpoint_0014999.pth', dataset=None):
    experiment = Path('../outputs/') / ckpt_path
    args = SimpleNamespace(config_file='configs/maskformer/maskformer_R50_bs16_160k_dino.yaml', opts=["GWM.DATASET", dataset], wandb_sweep_mode=False, resume_path=str(experiment / f'checkpoints/{checkpoint_num}'), eval_only=True)
    cfg = setup(args)
    random_state = ut.random_state.PytorchRNGState(seed=cfg.SEED).to(cfg.MODEL.DEVICE)   # .to(torch.device('cuda:0'))   

    model = Trainer.build_model(cfg)
    checkpointer = DetectionCheckpointer(model,
                                         random_state=random_state,
                                         save_dir=os.path.join(cfg.OUTPUT_DIR, '../..', 'checkpoints'))

    checkpoint_path = str(experiment / f'checkpoints/{checkpoint_num}')    # TODO change this! checkpoint_best.pth -> checkpoint_0009999.pth
    print(f'loading checkpoint from: {checkpoint_path}')
    checkpoint = checkpointer.resume_or_load(checkpoint_path, resume=False)
    model.eval()
    return model, cfg


def iou_np(mask, gt, thres=0.5):
    mask = (mask >= thres).astype(float)
    intersect = (mask*gt).sum()
    union = mask.sum() + gt.sum() - intersect
    iou = intersect/union.clip(min=1e-12)
    return iou


if __name__=="__main__":
################################### pred: FBMS59 ######################################################

    DATASET = "FBMS"
    model1, cfg1 = load_model_cfg(f'fbms/fbms_zero_shot', 'checkpoint.pth', DATASET)
    binary_dir = f'../results/fbms'
    
    train_loader, val_loader = config.loaders(cfg1)
    for idx, sample in enumerate(tqdm(train_loader)):
        preds1 = model1.forward_base(sample, keys=cfg1.GWM.SAMPLE_KEYS, get_eval=True)
        masks_raw1 = torch.stack([x['sem_seg'] for x in preds1], 0)
        masks_softmaxed = torch.sigmoid(masks_raw1)

        gt_seg = torch.stack([x['sem_seg_ori'] for x in sample]).cpu()
        HW = gt_seg.shape[-2:]
        masks_upsampled = F.interpolate(masks_softmaxed.detach().cpu(), size=HW, mode='bilinear', align_corners=False)[0][0]
        masks_np = masks_upsampled.numpy()

        masks_np[masks_np >= 0.5] = 1.
        masks_np[masks_np < 0.5] = 0.
        masks_np = (masks_np * 255.).astype(np.uint8)

        dirname = sample[0]["dirname"]
        fname = sample[0]['fname']

        path_to_binary_mask = os.path.join(binary_dir, dirname, fname)
        os.makedirs(f"{binary_dir}/{dirname}", exist_ok=True)
        Image.fromarray(masks_np).save(path_to_binary_mask)
    
 
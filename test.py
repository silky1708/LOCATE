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
    # experiment = Path('../checkpoints/') / ckpt_path
    
    # args = SimpleNamespace(config_file=str(experiment / 'config.yaml'), opts=[], wandb_sweep_mode=False, resume_path=str(experiment / 'checkpoints/checkpoint_best.pth'), eval_only=True)  # better way 
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
########################## computing metrics for unsupervised image segmentation #############################
    # datasets: CUB, OMRON, DUTS, ECSSD, FLOWERS-102 
    
    # for ckpt_num in tqdm(range(5000, 100000, 5000)):
    #     with open("image_metrics.json", 'r') as fp:
    #         metrics = json.load(fp)
    #     if DATASET not in metrics:
    #         metrics[DATASET] = {}

    # DATASET = "INTERNET"
    # ckpt_num = 25000
    # results_dir = '../../rand_web_imgs/locate_predictions'

    # checkpoint_num = f'checkpoint_{str(ckpt_num-1).zfill(7)}.pth'
    # # print(f'{"*"*10} using checkpoint... {checkpoint_num}')
    # ckpt_dirname = 'davis+stv2+fbms/20230429_061157'

    # if os.path.exists(f'../outputs/{ckpt_dirname}/checkpoints/{checkpoint_num}'):
    #     model, cfg = load_model_cfg(ckpt_dirname, checkpoint_num, DATASET)
    #     _, val_loader = config.loaders(cfg)
        
    #     outputs = []
    #     for idx, sample in enumerate(tqdm(val_loader)):
    #         with torch.no_grad():
    #             preds = model.forward_base(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True)
    #             masks_raw = torch.stack([x['sem_seg'] for x in preds], 0)
    #             masks_softmaxed = torch.sigmoid(masks_raw)
                
    #             # gt_seg = torch.stack([x['sem_seg_ori'] for x in sample]).cpu()[0]
    #             # HW = gt_seg.shape[-2:]
    #             HW = sample[0]["orig_size"]
    #             masks_upsampled = F.interpolate(masks_softmaxed.detach().cpu(), size=HW, mode='bilinear', align_corners=False)[0]   # 1xHxW
    #             # results = compute_metrics(
    #             #             targets=gt_seg,
    #             #             preds=masks_upsampled,
    #             #             metrics=['acc', 'iou', 'f_beta', 'f_max'],
    #             #             threshold=0.5,
    #             #             swap_dims=False)
    #             # outputs.append(results)

    #             masks_np = masks_upsampled[0].numpy()
    #             masks_np[masks_np >= 0.5] = 1.
    #             masks_np[masks_np < 0.5] = 0.
    #             masks_np = (masks_np*255.).astype(np.uint8)

    #             fname = sample[0]["fname"]
    #             # dirname = sample[0]["dirname"]
    #             save_path = os.path.join(results_dir)
    #             os.makedirs(save_path, exist_ok=True)
    #             Image.fromarray(masks_np).save(os.path.join(save_path, fname))


        # outputs = list_of_dict_of_lists_to_dict_of_lists(outputs)
        # results = aggregate_metrics(outputs)
        # print(results)

        # metrics[DATASET][checkpoint_num] = results

        # json_obj = json.dumps(metrics, indent=2)
        # with open('image_metrics.json', 'w') as fp:
        #     fp.write(json_obj)



################################### pred: FBMS59 ######################################################

    DATASET = "FBMS"
    # with open('./video_metrics.json', 'r') as fp:
    #     video_metrics = json.load(fp)
    # print('already found..\n', video_metrics)

    # if DATASET not in video_metrics:
    #     video_metrics[DATASET] = {}

    # for ckpt_num in tqdm(range(5000, 100000, 5000)):
    #     checkpoint_num = f'checkpoint_{str(ckpt_num-1).zfill(7)}.pth'

    # if os.path.exists(f'../checkpoints/DAVIS/checkpoints/checkpoint_best.pth'):
    # frame_avg = 0.
    # total_files = 0
    # gt_dir = '../../FBMS59_clean/Annotations'

    model1, cfg1 = load_model_cfg(f'fbms/fbms_zero_shot_25k_iter6', 'checkpoint_0024999.pth', DATASET)
    binary_dir = f'../results/fbms_zero_shot_25k_iter6'    #  f'../../FBMS59_clean/locate_predictions'
    # binary_dir = os.path.join(results_dir, 'binary_dir') 
    
    train_loader, val_loader = config.loaders(cfg1)
    for idx, sample in enumerate(tqdm(train_loader)):
        # total_files += 1
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
        fname = sample[0]['fname']      # sample[0]["fname"]      # {video_name}_xxxxx.jpg 
        # if dirname == "tennis":
        #     fnum = int(fname.split('.')[0][6:])
        # else:
        #     fnum = int(fname.split('.')[0].split('_')[-1])

        path_to_binary_mask = os.path.join(binary_dir, dirname, fname)
        os.makedirs(f"{binary_dir}/{dirname}", exist_ok=True)
        Image.fromarray(masks_np).save(path_to_binary_mask)

            # gt_img = Image.open(os.path.join(gt_dir, dirname, fname))
            #     iou = iou_np(masks_np, gt_seg[0][0].numpy())     # iou_np(masks_np, np.array(gt_img)/255.)
            #     frame_avg += iou
            
            # print('using checkpoint...', checkpoint_num)
            # print('total files:', total_files)
            # print('frame avg:', round(frame_avg/total_files, 2), frame_avg/total_files)

            # miou = (frame_avg/total_files) * 100.
            # video_metrics[DATASET][checkpoint_num] = miou
            # json_obj = json.dumps(video_metrics, indent=2)
            # with open('./video_metrics.json', 'w') as fp:
            #     fp.write(json_obj)
    
    
#################################### pred - SegTrackv2 ##################################################
    # model1, cfg1 = load_model_cfg(f'stv2/stv2_20k_iter1', 'checkpoint_0004999.pth',  "STv2")

    # # frame_avg = 0.
    # # total_files = 0
    # # gt_dir = '../../SegTrackv2/GroundTruth'

    # results_dir = '../../SegTrackv2/locate_predictions/binary_dir'
    # train_loader, val_loader = config.loaders(cfg1)
    # for idx, sample in enumerate(tqdm(val_loader)):
    #     # total_files += 1
    #     preds1 = model1.forward_base(sample, keys=cfg1.GWM.SAMPLE_KEYS, get_eval=True)
    #     masks_raw1 = torch.stack([x['sem_seg'] for x in preds1], 0)
    #     masks_softmaxed = torch.sigmoid(masks_raw1)
        
    #     gt_seg = torch.stack([x['sem_seg_ori'] for x in sample]).cpu()
    #     HW = gt_seg.shape[-2:]
    #     masks_upsampled = F.interpolate(masks_softmaxed.detach().cpu(), size=HW, mode='bilinear', align_corners=False)[0][0]
    #     masks_np = masks_upsampled.numpy()
    #     masks_np[masks_np >= 0.5] = 1.
    #     masks_np[masks_np < 0.5] = 0.
    #     masks_np = (masks_np*255.).astype(np.uint8)
        
    #     dirname = sample[0]["dirname"]
    #     fname = sample[0]["fname"]   # xxx.bmp/jpg/png
        
    #     save_dir_path = os.path.join(results_dir, dirname)
    #     os.makedirs(save_dir_path, exist_ok=True)
    #     Image.fromarray(masks_np).save(os.path.join(save_dir_path, fname))


        # gt_img = Image.open(os.path.join(gt_dir, dirname, fname))
        # iou = iou_np(masks_np, np.array(gt_img)[:,:,0]/255.)
        # frame_avg += iou
     
    # print('total files:', total_files)
    # print('frame avg:', frame_avg/total_files)
        
            
    # ################### prediction of the best checkpoint - DAVIS16 ################################

    # DATASET = "DAVIS"
    # with open('./video_metrics.json', 'r') as fp:
    #     video_metrics = json.load(fp)
    # print('already found..\n', video_metrics)

    # if DATASET not in video_metrics:
    #     video_metrics[DATASET] = {}

    # for ckpt_num in tqdm(range(5000, 100000, 5000)):
    #     checkpoint_num = f'checkpoint_{str(ckpt_num-1).zfill(7)}.pth'

    # if os.path.exists(f'../outputs/davis/davis_20k_iter8/checkpoints/checkpoint_best.pth'):
    #     model1, cfg1 = load_model_cfg(f'davis/davis_20k_iter8', 'checkpoint_0019999.pth', DATASET)

    # if os.path.exists(f'../checkpoints/DAVIS/checkpoints/checkpoint_best.pth'):
    #     model1, cfg1 = load_model_cfg(f'DAVIS', 'checkpoint_best.pth', DATASET)

    ## TODO put train loader's batch size = 1. before running test.py 

    ## results_dir = f'../../DAVIS/locate_predictions'
    ## sigmoid_dir = os.path.join(results_dir, 'sigmoid_dir')
    ## binary_dir = os.path.join(results_dir, 'binary_dir')

    # train_loader, val_loader = config.loaders(cfg1)
    # F_scores = {}
    # gwm_pred_dir = '../../DAVIS16_OCLR/DAVIS16_test_adap_with_crf'    #f'../../guess-what-moves-new/gwm_predictions/DAVIS'

    # for idx, sample in enumerate(tqdm(val_loader)):
    #     # preds1 = model1.forward_base(sample, keys=cfg1.GWM.SAMPLE_KEYS, get_eval=True)
    #     # masks_raw1 = torch.stack([x['sem_seg'] for x in preds1], 0)
    #     # masks_softmaxed = torch.sigmoid(masks_raw1)

    #     gt_seg = torch.stack([x['sem_seg_ori'] for x in sample]).cpu()
        
    #     # HW = gt_seg.shape[-2:]
    #     # masks_upsampled = F.interpolate(masks_softmaxed.detach().cpu(), size=HW, mode='bilinear', align_corners=False)[0]  # 1xHxW
    #     # masks_upsampled[masks_upsampled >= 0.5] = 1.
    #     # masks_upsampled[masks_upsampled < 0.5] = 0.

    #     dirname = sample[0]['dirname']
    #     fname = sample[0]['fname']

    #     gwm_pred = np.array(Image.open(os.path.join(gwm_pred_dir, dirname, f'{fname}.png')))/255.
    #     gwm_pred = torch.from_numpy(gwm_pred).unsqueeze(0)   # 1xHxW
    #     # print('gwm_pred shape:', gwm_pred.shape, 'gt shape:', gt_seg[0].shape)

    #     p,r = precision_recall(gt_seg[0], gwm_pred)
    #     f_score = F_vanilla(p, r)

    #     F_scores[dirname] = F_scores.get(dirname, [])
    #     F_scores[dirname].append(f_score)
        

    # for dirname, fscores in F_scores.items():
    #     F_scores[dirname] = sum(fscores)/len(fscores)

    # assert len(F_scores) == 20, "val dirs not equal to 20! Check.."
    # seq_avg = sum(F_scores.values())/len(F_scores) 
    # print(f'mean F score: {seq_avg}')
    





        # dirname = sample[0]['dirname']
        # fname = f"{sample[0]['fname']}.png"

        # masks_np = masks_upsampled.numpy()[0][0]   # HxW
        # masks_np[masks_np >= 0.5] = 1.
        # masks_np[masks_np < 0.5] = 0.

    #         masks_binary = (masks_np*255.).astype(np.uint8)
    #         path_to_binary_mask = os.path.join(binary_dir, dirname, fname)
    #         os.makedirs(f"{binary_dir}/{dirname}", exist_ok=True)
    #         Image.fromarray(masks_binary).save(path_to_binary_mask)

            # iou = iou_np(masks_np, gt_seg[0][0].numpy())
            # iou_scores[dirname] = iou_scores.get(dirname, [])
            # iou_scores[dirname].append(iou)


        # for dirname, ious in iou_scores.items():
        #     iou_scores[dirname] = sum(ious)/len(ious)
        
        # assert len(iou_scores) == 20, "val dirs not equal to 20! Check.."
        # seq_avg = sum(iou_scores.values())/len(iou_scores) 

        # video_metrics[DATASET][checkpoint_num] = seq_avg * 100.
        # json_obj = json.dumps(video_metrics, indent=2)
        # with open('./video_metrics.json', 'w') as fp:
        #     fp.write(json_obj)


        #         masks_sigmoid = (masks_np*255.).astype(np.uint8)
        #         path_to_sigmoid_mask = os.path.join(sigmoid_dir, dirname, f"{fname}.png")
        #         os.makedirs(f"{sigmoid_dir}/{dirname}", exist_ok=True)
        #         Image.fromarray(masks_sigmoid).save(path_to_sigmoid_mask)





# #################### crf post processing.. does not help!!  #####################
    # pred_dir = '../results/davis_20k_iter4/binary_dir'
    # img_dir = '../../DAVIS/JPEGImages/480p'
    # gt_dir = '../../DAVIS/Annotations/480p'

    # transform_rgb = transforms.Compose([transforms.ToTensor(), 
    #                 transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

    # with open('../results/val.txt', 'r') as fp:
    #     val_videos = fp.readlines()
        
    # miou_before_crf = 0.
    # miou_after_crf = 0.
    # for val_vid in tqdm(val_videos):
    #     val_vid = val_vid.strip()    # bear
        
    #     vid_dir_path = os.path.join(img_dir, val_vid)
    #     val_files = sorted(glob.glob(f"{vid_dir_path}/*.jpg"))[1:]
        
    #     iou_before_crf = 0.
    #     iou_after_crf = 0.
    #     for val_file in tqdm(val_files):
    #         img = Image.open(val_file).convert("RGB")
    #         img_t = transform_rgb(img)
            
    #         fname = val_file.split('/')[-1].split('.')[0]
    #         pred = np.array(Image.open(os.path.join(pred_dir, val_vid, f"{fname}.png")))/255.
    #         pred_t = torch.from_numpy(pred).unsqueeze(0)      # 1xHxW
    #         pred_t = torch.cat([1. - pred_t, pred_t], dim=0)   # 2xHxW
            
    #         pred_crf = dense_crf(img_t, pred_t).argmax(0)  # np.ndarray, HxW
            
    #         gt = np.array(Image.open(os.path.join(gt_dir, val_vid, f"{fname}.png")))/255.
            
    #         iou_before_crf += iou_np(pred, gt)
    #         iou_after_crf += iou_np(pred_crf, gt)
            
    #     iou_before_crf /= len(val_files)
    #     iou_after_crf /= len(val_files)
        
    #     print(f"{val_vid}, len: {len(val_files)}, before crf: {iou_before_crf}, after crf: {iou_after_crf}")
        
    #     miou_before_crf += iou_before_crf
    #     miou_after_crf += iou_after_crf
        
    # miou_before_crf /= len(val_videos)
    # miou_after_crf /= len(val_videos)
    # print(f"({len(val_videos)}) | miou: before crf -> {miou_before_crf}, after crf -> {miou_after_crf}")



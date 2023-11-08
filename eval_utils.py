import functools
import random
from collections import defaultdict

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

import flow_reconstruction
from utils import visualisation, log, grid
from utils.vit_extractor import ViTExtractor

label_colors = visualisation.create_label_colormap()
logger = log.getLogger('gwm')


def __default_font(fontsize):
    try:
        FNT = ImageFont.truetype("dejavu/DejaVuSansMono.ttf", fontsize)
    except OSError:
        FNT = ImageFont.truetype("dejavu/DejaVuSans.ttf", fontsize)
    return FNT


@functools.lru_cache(None)  # cache the result
def autosized_default_font(size_limit: float) -> ImageFont.ImageFont:
    fontsize = 1  # starting font size
    font = __default_font(fontsize)
    while font.getsize('test123')[1] < size_limit:
        fontsize += 1
        font = __default_font(fontsize)
    fontsize -= 1
    font = __default_font(fontsize)
    return font


def iou(masks, gt, thres=0.5):
    masks = (masks >= thres).float()
    intersect = torch.tensordot(masks, gt, dims=([-2, -1], [-2, -1]))   # Nx1; gt -> [0, 1]
    union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
    return intersect / union.clip(min=1e-12)


def get_unsup_image_viz(model, cfg, sample, criterion):
    if model.training:
        model.eval()
        preds = model.forward_base(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True)
        model.train()
    else:
        preds = model.forward_base(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True)
    return get_image_vis(model, cfg, sample, preds, criterion)

def get_vis_header(header_size, image_size, header_texts, header_height=20):
    W, H = (image_size, header_height)
    header_labels = []
    font = autosized_default_font(0.8 * H)

    for text in header_texts:
        im = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(im)
        w, h = draw.textsize(text, font=font)
        draw.text(((W - w) / 2, (H - h) / 2), text, fill="black", font=font)
        header_labels.append(torch.from_numpy(np.array(im)))
    header_labels = torch.cat(header_labels, dim=1)
    ret = (torch.ones((header_height, header_size, 3)) * 255)
    ret[:, :header_labels.size(1)] = header_labels

    return ret.permute(2, 0, 1).clip(0, 255).to(torch.uint8)

def get_image_vis(model, cfg, sample, preds, criterion):
    masks_pred = torch.stack([x['sem_seg'] for x in preds], 0)

    with torch.no_grad():
        flow = torch.stack([x['flow'].to(model.device) for x in sample]).clip(-20, 20)

    masks_softmaxed = torch.softmax(masks_pred, dim=1)
    masks_pred = masks_softmaxed
    rec_flows = criterion.flow_reconstruction(sample, criterion.process_flow(sample, flow), masks_softmaxed)
    rec_headers = ['rec_flow']
    if len(rec_flows) > 1:
        rec_headers.append('rec_bwd_flow')

    rgb = torch.stack([x['rgb'] for x in sample])
    flow = criterion.viz_flow(criterion.process_flow(sample, flow).cpu()) * 255
    rec_flows = [
        (criterion.viz_flow(rec_flow_.detach().cpu().cpu()) * 255).clip(0, 255).to(torch.uint8) for rec_flow_ in rec_flows
    ]


    gt_labels = torch.stack([x['sem_seg'] for x in sample])
    gt = F.one_hot(gt_labels, gt_labels.max().item() + 1).permute(0, 3, 1, 2)
    target_K = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
    masks = F.one_hot(masks_pred.argmax(1).cpu(), target_K).permute(0, 3, 1, 2)
    masks_each = torch.stack([masks_softmaxed, masks_softmaxed, masks_softmaxed], 2) * 255
    masks_each = einops.rearrange(F.pad(masks_each.cpu(), pad=[0, 1], value=255), 'b n c h w -> b c h (n w)')

    gt_seg = torch.einsum('b k h w, k c -> b c h w', gt, label_colors[:gt_labels.max().item() + 1])
    pred_seg = torch.einsum('b k h w, k c -> b c h w', masks, label_colors[:target_K])
    if all('gwm_seg' in d for d in sample):
        gwm_labels = torch.stack([x['gwm_seg'] for x in sample])
        mg = F.one_hot(gwm_labels, gwm_labels.max().item() + 1).permute(0, 3, 1, 2)
        gwm_seg = torch.einsum('b k h w, k c -> b c h w', mg, label_colors[:gwm_labels.max().item() + 1])
        image_viz = torch.cat(
            [rgb, flow, F.pad(gt_seg.cpu(), pad=[0, 1], value=255), F.pad(gwm_seg, pad=[0, 1], value=255),
             pred_seg.cpu(), *rec_flows], -1)
        header_text = ['rgb', 'gt_flow', 'gt_seg', 'GWM', 'pred_seg', *rec_headers]
    else:
        image_viz = torch.cat([rgb, flow, gt_seg.cpu(), pred_seg.cpu(), *rec_flows], -1)
        header_text = ['rgb', 'gt_flow', 'gt_seg', 'pred_seg', *rec_headers]

    image_viz = torch.cat([image_viz, masks_each], -1)
    header_text.extend(['slot'] * masks_softmaxed.shape[1])
    if 'flow_edges' in sample[0]:
        flow_edges = torch.stack([x['flow_edges'].to(image_viz.device) for x in sample])
        if len(flow_edges.shape) >= 4:
            flow_edges = flow_edges.sum(1, keepdim=len(flow_edges.shape) == 4)
        flow_edges = flow_edges.expand(-1, 3, -1, -1)
        flow_edges = flow_edges * 255
        image_viz = torch.cat([image_viz, flow_edges], -1)
        header_text.append('flow_edges')
    image_viz = einops.rearrange(image_viz[:8], 'b c h w -> c (b h) w').detach().clip(0, 255).to(torch.uint8)

    return image_viz, header_text


def get_frame_vis(model, cfg, sample, preds):
    masks_pred = torch.stack([x['sem_seg'] for x in preds], 0)
    flow = torch.stack([x['flow'].to(model.device) for x in sample]).clip(-20, 20)

    masks_softmaxed = torch.softmax(masks_pred, dim=1)
    if cfg.GWM.SIMPLE_REC:
        mask_denom = einops.reduce(masks_softmaxed, 'b k h w -> b k 1', 'sum') + 1e-7
        means = torch.einsum('brhw, bchw -> brc', masks_softmaxed, flow) / mask_denom
        rec_flow = torch.einsum('bkhw, bkc-> bchw', masks_softmaxed, means)
    elif cfg.GWM.HOMOGRAPHY:
        rec_flow = flow_reconstruction.get_quad_flow(masks_softmaxed, flow)
    else:
        grid_x, grid_y = grid.get_meshgrid(cfg.GWM.RESOLUTION, model.device)
        rec_flow = flow_reconstruction.get_quad_flow(masks_softmaxed, flow, grid_x, grid_y)

    rgb = torch.stack([x['rgb'] for x in sample])
    flow = torch.stack([visualisation.flow2rgb_torch(x) for x in flow.cpu()]) * 255
    rec_flow = torch.stack([visualisation.flow2rgb_torch(x) for x in rec_flow.detach().cpu()]) * 255

    gt_labels = torch.stack([x['sem_seg'] for x in sample])
    gt = F.one_hot(gt_labels, gt_labels.max().item() + 1).permute(0, 3, 1, 2)

    masks = F.one_hot(masks_pred.argmax(1).cpu(), cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES).permute(0, 3, 1, 2)

    gt_seg = torch.einsum('b k h w, k c -> b c h w', gt, label_colors[:gt_labels.max().item() + 1])
    pred_seg = torch.einsum('b k h w, k c -> b c h w', masks, label_colors[:cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES])
    frame_vis = torch.cat([rgb, flow, gt_seg.cpu(), pred_seg.cpu(), rec_flow.clip(0, 255).to(torch.uint8)], -1)
    frame_vis = einops.rearrange(frame_vis, 'b c h w -> b c h w').detach().clip(0, 255).to(torch.uint8)
    return frame_vis


def is_2comp_dataset(dataset):
    if '+' in dataset:
        d = dataset.split('+')[0].strip()
    else:
        d = dataset.strip()
    logger.info_once(f"Is 2comp dataset? {d}")
    for s in ['DAVIS', 'FBMS', 'STv2']:
        if s in d:
            return True
    return d in ['DAVIS',
                 'FBMS',
                 'STv2']

def eval_unsupmf(cfg, val_loader, model, criterion, writer=None, writer_iteration=0, use_wandb=False):
    logger.info(f'Running Evaluation: {cfg.LOG_ID} {"Simple" if cfg.GWM.SIMPLE_REC else "Gradient"}:')
    logger.info(f'Model mode: {"train" if model.training else "eval"}, wandb: {use_wandb}')
    logger.info(f'Dataset: {cfg.GWM.DATASET} # components: {cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES}')

    # merger = None
    # if cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES > 2:
    #     merger = MaskMerger(cfg, model)

    print_idxs = random.sample(range(len(val_loader)), k=10)

    images_viz = []
    ious_davis_eval = defaultdict(list)
    ious = defaultdict(list)  # deafultdict never raises a KeyError; here list(empty?) will be returned if `key` is not present.
    
    frame_mean_iou = 0.
    for idx, sample in enumerate(tqdm(val_loader)):
        t = 1
        # sample = [e for s in sample for e in s]
        # category = [s['category'] for s in sample]
        preds = model.forward_base(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True)
        masks_raw = torch.stack([x['sem_seg'] for x in preds], 0)

        masks_softmaxed = torch.sigmoid(masks_raw)  # torch.softmax(masks_raw, dim=1)
        masks_dict = masks_softmaxed   # merger(sample, masks_softmaxed)

        # if writer and idx in print_idxs:
        #     flow = torch.stack([x['flow'] for x in sample]).to(model.device)
        #     img_viz, header_text = get_image_vis(model, cfg, sample, preds, criterion)
        #     images_viz.append(img_viz)

        masks = masks_dict   # masks_dict['cos']
        gt_seg = torch.stack([x['sem_seg_ori'] for x in sample]).cpu()
        HW = gt_seg.shape[-2:]
        if HW != masks.shape[-2:]:
            logger.info_once(f"Upsampling predicted masks to {HW} for evaluation")
            masks_softmaxed_sel = F.interpolate(masks.detach().cpu(), size=HW, mode='bilinear', align_corners=False)
        else:
            masks_softmaxed_sel = masks.detach().cpu()
        
        frame_mean_iou += iou(masks_softmaxed_sel, gt_seg.float()).squeeze().item()
        
    frame_mean_iou /= len(val_loader)
    
#############################         DAVIS16 dataset ################################
#         dirname = sample[0]['dirname']
#         ious_davis_eval[dirname].append(iou(masks_softmaxed_sel, gt_seg.float()).squeeze().item())
        
#     for dname, iou_scores in ious_davis_eval.items():
#         ious_davis_eval[dname] = sum(iou_scores)/len(iou_scores)

#         frame_mean_iou = sum(ious_davis_eval.values())/len(ious_davis_eval) * 100.
#         if writer:
#             writer.add_scalar('eval/mIoU', frame_mean_iou, writer_iteration)

    logger.info(f"mIoU: {round(frame_mean_iou*100., 2)} \n")
    return frame_mean_iou
    
    
        # masks_ = einops.rearrange(masks_softmaxed_sel, '(b t) s h w -> b t s 1 h w', t=t).detach()
        # gt_seg = einops.rearrange(gt_seg, 'b h w -> b 1 h w').float()
        # for i in range(masks_.size(0)):
        #     masks_k = F.interpolate(masks_[i], size=(1, gt_seg.shape[-2], gt_seg.shape[-1]))  # t s 1 h w
        #     mask_iou = iou(masks_k[:, :, 0], gt_seg[i, 0], thres=0.5)  # t s
        #     iou_max, slot_max = mask_iou.max(dim=1)

        #     ious[category[i][0]].append(iou_max)
        #     frame_id = category[i][1]
        #     ious_davis_eval[category[i][0]].append((frame_id.strip().replace('.png', ''), iou_max))

    # frameious = sum(ious.values(), [])
    # frame_mean_iou = torch.cat(frameious).sum().item() * 100 / len(frameious)
    # if 'DAVIS' in cfg.GWM.DATASET.split('+')[0]:
    #     logger.info_once("Using DAVIS evaluator methods for evaluting IoU -- mean of mean of sequences without first frame")
    #     seq_scores = dict()
    #     for c in ious_davis_eval:
    #         seq_scores[c] = np.nanmean([v.item() for n, v in ious_davis_eval[c] if int(n) > 1])

    #     frame_mean_iou = np.nanmean(list(seq_scores.values())) * 100

                # header = get_vis_header(images_viz[0].size(2), flow.size(3), header_text)
            # images_viz = torch.cat(images_viz, dim=1)
            # images_viz = torch.cat([header, images_viz], dim=1)
            # writer.add_image('val/images', images_viz, writer_iteration)  # CxHxW
    




class MaskMerger:
    def __init__(self, cfg, model, merger_model="dino_vits8"):
        self.extractor = ViTExtractor(model_type=merger_model, device=model.device)
        self.out_dim = 384

        self.mu = torch.tensor(self.extractor.mean).to(model.device).view(1, -1, 1, 1)
        self.sigma = torch.tensor(self.extractor.std).to(model.device).view(1, -1, 1, 1)
        self.start_idx = 0

    def get_feats(self, batch):
        with torch.no_grad():
            feat = self.extractor.extract_descriptors(batch, facet='key', layer=11, bin=False)
            feat = feat.reshape(feat.size(0), *self.extractor.num_patches, -1).permute(0, 3, 1, 2)
            return F.interpolate(feat, batch.shape[-2:], mode='bilinear')

    def spectral(self, A):
        clustering = SpectralClustering(n_clusters=2,
                                        affinity='precomputed',
                                        random_state=0).fit(A.detach().cpu().numpy())
        return np.arange(A.shape[-1])[clustering.labels_ == 0], np.arange(A.shape[-1])[clustering.labels_ == 1]

    def cos_merge(self, basis, masks):
        basis = basis / torch.linalg.vector_norm(basis, dim=-1, keepdim=True).clamp(min=1e-6)
        A = torch.einsum('brc, blc -> brl', basis, basis)[0].clamp(min=1e-6)
        inda, indb = self.spectral(A)
        return torch.stack([masks[:, inda].sum(1),
                            masks[:, indb].sum(1)], 1)

    def __call__(self, sample, masks_softmaxed):
        with torch.no_grad():
            masks_softmaxed = masks_softmaxed[:, self.start_idx:]
            batch = torch.stack([x['rgb'].to(masks_softmaxed.device) for x in sample], 0) / 255.0
            features = self.get_feats((batch - self.mu) / self.sigma)
            basis = torch.einsum('brhw, bchw -> brc', masks_softmaxed, features)
            basis /= einops.reduce(masks_softmaxed, 'b r h w -> b r 1', 'sum').clamp_min(1e-12)

            return {
                'cos': self.cos_merge(basis, masks_softmaxed),
            }



# from https://github.com/lukemelas/unsupervised-image-segmentation/
@torch.no_grad()
def compute_metrics(preds, targets, metrics=['f_max', 'acc', 'iou'], threshold=0.5, swap_dims=False):

    # Move to CPU
    preds = preds.detach()  # .cpu()
    targets = targets.detach()  # .cpu()
    # assert len(preds.shape) == 4 and len(targets.shape) == 3
    # soft_preds = torch.softmax(preds, dim=1)[:, (0 if swap_dims else 1)]  # convert to probabilities
    soft_preds = preds
    hard_preds = soft_preds > threshold

    # Compute 
    results = defaultdict(list)
    for soft_pred, hard_pred, target in zip(soft_preds, hard_preds, targets):
        if 'f_max' in metrics:
            precision, recall = compute_prs(soft_pred, target, prob_bins=255)
            results['f_max_precision'].append(precision)
            results['f_max_recall'].append(recall)
        if 'f_beta' in metrics:
            precision, recall = precision_recall(target, hard_preds)
            results['f_beta_precision'].append([precision])
            results['f_beta_recall'].append([recall])
        if 'acc' in metrics:
            acc = compute_accuracy(hard_pred, target)
            results['acc'].append(acc)
        if 'iou' in metrics:
            iou = compute_iou(hard_pred, target)
            results['iou'].append(iou)
    return results


@torch.no_grad()
def aggregate_metrics(totals):
    results = defaultdict(list)
    if 'acc' in totals:
        results['acc'] = mean(totals['acc'])
    if 'iou' in totals:
        results['iou'] = mean(totals['iou'])
    if 'loss' in totals:
        results['loss'] = mean(totals['loss'])
    if 'f_max_precision' in totals and 'f_max_recall' in totals:
        precisions = torch.tensor(totals['f_max_precision'])
        recalls = torch.tensor(totals['f_max_recall'])
        results['f_max'] = F_max(precisions, recalls)
    if 'f_beta_precision' in totals and 'f_beta_recall' in totals:
        precisions = torch.tensor(totals['f_beta_precision'])
        recalls = torch.tensor(totals['f_beta_recall'])
        results['f_beta'] = F_max(precisions, recalls)
    return results


def compute_accuracy(pred, target):
    pred, target = pred.to(torch.bool), target.to(torch.bool)
    return torch.mean((pred == target).to(torch.float)).item()


def compute_iou(pred, target):
    pred, target = pred.to(torch.bool), target.to(torch.bool)
    intersection = torch.sum(pred * (pred == target), dim=[-1, -2]).squeeze()
    union = torch.sum(pred + target, dim=[-1, -2]).squeeze()
    iou = (intersection.to(torch.float) / union).mean()
    iou = iou.item() if (iou == iou) else 0  # deal with nans, i.e. torch.nan_to_num(iou, nan=0.0)
    return iou


def compute_prs(pred, target, prob_bins=255):
    p = []
    r = []
    for split in np.arange(0.0, 1.0, 1.0 / prob_bins):
        if split == 0.0:
            continue
        pr = precision_recall(target, pred > split)
        p.append(pr[0])
        r.append(pr[1])
    return p, r


def precision_recall(mask_gt, mask):
    mask_gt, mask = mask_gt.to(torch.bool), mask.to(torch.bool)
    true_positive = torch.sum(mask_gt * (mask_gt == mask), dim=[-1, -2]).squeeze()
    mask_area = torch.sum(mask, dim=[-1, -2]).to(torch.float)
    mask_gt_area = torch.sum(mask_gt, dim=[-1, -2]).to(torch.float)
    precision = true_positive / mask_area
    precision[mask_area == 0.0] = 1.0
    recall = true_positive / mask_gt_area
    recall[mask_gt_area == 0.0] = 1.0
    return precision.item(), recall.item()


def F_scores(p, r, betta_sq=0.3):
    f_scores = ((1 + betta_sq) * p * r) / (betta_sq * p + r)
    f_scores[f_scores != f_scores] = 0.0   # handle nans
    return f_scores


def F_max(precisions, recalls, betta_sq=0.3):
    f_scores = F_scores(precisions, recalls, betta_sq)
    f_scores = f_scores.mean(dim=0)
    return f_scores.max().item()

def F_vanilla(precision, recall):
    return (2.*precision*recall)/(precision + recall)


def mean(x):
    return sum(x) / len(x)


def list_of_dicts_to_dict_of_lists(LD):
    return {k: [dic[k] for dic in LD] for k in LD[0]}


def list_of_dict_of_lists_to_dict_of_lists(LD):
    return {k: [v for dic in LD for v in dic[k]] for k in LD[0]}


def dict_of_lists_to_list_of_dicts(DL):
    return [dict(zip(DL, t)) for t in zip(*DL.values())]
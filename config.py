import copy
import itertools
import logging
import os
from pathlib import Path

import numpy as np
import torch.utils.data
from detectron2.config import CfgNode as CN

import utils
from datasets import FlowPairDetectron, FlowEvalDetectron, DavisTrainDataset, DavisValDataset, STv2Dataset, FBMSTrainDataset, FBMSValDataset, AllVideoTrainDataset, CUBTest, DUTSTest, ECSSDTest, OMRONTest, FlowersTest, RandomInternetImages

logger = logging.getLogger('gwm')

def scan_train_flow(folders, res, pairs, basepath):
    pair_list = [p for p in itertools.combinations(pairs, 2)]

    flow_dir = {}
    for pair in pair_list:
        p1, p2 = pair
        flowpairs = []
        for f in folders:
            path1 = basepath / f'Flows_gap{p1}' / res / f
            path2 = basepath / f'Flows_gap{p2}' / res / f

            flows1 = [p.name for p in path1.glob('*.flo')]
            flows2 = [p.name for p in path2.glob('*.flo')]

            flows1 = sorted(flows1)
            flows2 = sorted(flows2)

            intersect = list(set(flows1).intersection(flows2))
            intersect.sort()

            flowpair = np.array([[path1 / i, path2 / i] for i in intersect])
            flowpairs += [flowpair]
        flow_dir['gap_{}_{}'.format(p1, p2)] = flowpairs

    # flow_dir is a dictionary, with keys indicating the flow gap, and each value is a list of sequence names,
    # each item then is an array with Nx2, N indicates the number of available pairs.
    return flow_dir


def setup_dataset(cfg=None, multi_val=False):
    dataset_str = cfg.GWM.DATASET
    if '+' in dataset_str:
        datasets = dataset_str.split('+')
        logger.info(f'Multiple datasets detected: {datasets}')
        train_datasets = []
        val_datasets = []
        for ds in datasets:
            proxy_cfg = copy.deepcopy(cfg)
            proxy_cfg.merge_from_list(['GWM.DATASET', ds]),
            train_ds, val_ds = setup_dataset(proxy_cfg, multi_val=multi_val)
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
        logger.info(f'Multiple datasets detected: {datasets}')
        logger.info(f'Validation is still : {datasets[0]}')
        return torch.utils.data.ConcatDataset(train_datasets), val_datasets[0]

    resolution = cfg.GWM.RESOLUTION  # h,w 
    res = ""
    with_gt = True
    pairs = [1, 2, -1, -2]
    trainval_data_dir = None

    if cfg.GWM.DATASET == 'DAVIS':
        basepath = 'DAVIS'
        img_dir = 'DAVIS/JPEGImages/480p'
        pseudo_gt_dir = 'DAVIS/graphcut_arflow_sintel_dinov2'           
        sem_seg_dir = 'DAVIS/Annotations/480p'

        # val_flow_dir = 'DAVIS/Flows_gap1'
        val_seq = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees',
                   'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane',
                   'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch']
        val_data_dir = [img_dir, pseudo_gt_dir, sem_seg_dir]
        train_data_dir = val_data_dir
        res = ""
        data_dirs = None

    elif cfg.GWM.DATASET in ['FBMS']:
        basepath = 'FBMS59_clean'
        train_img_dir = 'FBMS59'
        pseudo_gt_dir = 'guess-what-moves/results/fbms'
        sem_seg_dir = 'FBMS59_clean/Annotations'

        train_data_dir = [train_img_dir, pseudo_gt_dir, sem_seg_dir]

        val_seq = ['camel01', 'cars1', 'cars10', 'cars4', 'cars5', 'cats01', 'cats03', 'cats06',
                   'dogs01', 'dogs02', 'farm01', 'giraffes01', 'goats01', 'horses02', 'horses04',
                   'horses05', 'lion01', 'marple12', 'marple2', 'marple4', 'marple6', 'marple7', 'marple9',
                   'people03', 'people1', 'people2', 'rabbits02', 'rabbits03', 'rabbits04', 'tennis']
        
        val_img_dir = 'FBMS59_clean/JPEGImages'
        val_gt_dir = '/FBMS_val/Annotations'
        val_data_dir = [val_img_dir, pseudo_gt_dir, sem_seg_dir]
        with_gt = False
        pairs = [3, 6, -3, -6]
        data_dirs = None

    elif cfg.GWM.DATASET in ['STv2']:
        basepath = 'SegTrackv2'
        img_dir = 'SegTrackv2/JPEGImages'
        pseudo_gt_dir = 'guess-what-moves/results/stv2'     
        sem_seg_dir = 'SegTrackv2/GroundTruth'

        val_seq = ['drift', 'birdfall', 'girl', 'cheetah', 'worm', 'parachute', 'monkeydog',
                   'hummingbird', 'soldier', 'bmx', 'frog', 'penguin', 'monkey', 'bird_of_paradise']
        val_data_dir = [img_dir, pseudo_gt_dir, sem_seg_dir]
        train_data_dir = val_data_dir
        data_dirs = None

    elif cfg.GWM.DATASET in ["all"]:
        data_dirs = []

        img_dir = 'DAVIS/JPEGImages/480p'
        pseudo_gt_dir = 'DAVIS/graphcut_arflow_sintel_masks'     
        sem_seg_dir = 'DAVIS/Annotations/480p'
        davis_data_dir = [img_dir, pseudo_gt_dir, sem_seg_dir]
        data_dirs.append(davis_data_dir)

        img_dir = 'SegTrackv2/JPEGImages'
        pseudo_gt_dir = 'SegTrackv2/graphcut_arflow_sintel'
        sem_seg_dir = 'SegTrackv2/GroundTruth'
        stv2_data_dir = [img_dir, pseudo_gt_dir, sem_seg_dir]
        data_dirs.append(stv2_data_dir)

        train_img_dir = 'FBMS59'
        pseudo_gt_dir = 'FBMS59/graphcut_arflow_sintel_fbms59'
        sem_seg_dir = 'FBMS59_clean/Annotations'
        fbms_data_dir = [train_img_dir, pseudo_gt_dir, sem_seg_dir]
        data_dirs.append(fbms_data_dir)
        
        train_data_dir = []
        val_data_dir = []

    elif cfg.GWM.DATASET in ["CUB"]:
        data_dirs = None
        train_data_dir = []
        image_dir = 'CUB_200_2011/test_images'
        sem_seg_dir = 'CUB_200_2011/test_segmentations'
        val_data_dir = [image_dir, sem_seg_dir]

    elif cfg.GWM.DATASET in ["OMRON"]:
        data_dirs = None
        train_data_dir = []
        image_dir = 'DUT-OMRON/DUT-OMRON-image'
        sem_seg_dir = 'DUT-OMRON/pixelwiseGT-new-PNG'
        val_data_dir = [image_dir, sem_seg_dir]


    elif cfg.GWM.DATASET in ["DUTS"]:
        data_dirs = None
        train_data_dir = []
        image_dir = 'DUTS-TE/DUTS-TE-Image'
        sem_seg_dir = 'DUTS-TE/DUTS-TE-Mask'
        val_data_dir = [image_dir, sem_seg_dir]

    elif cfg.GWM.DATASET in ["ECSSD"]:
        data_dirs = None
        train_data_dir = []
        image_dir = 'ECSSD/images'
        sem_seg_dir = 'ECSSD/ground_truth_mask'
        val_data_dir = [image_dir, sem_seg_dir]

    elif cfg.GWM.DATASET in ["FLOWERS"]:
        data_dirs = None
        train_data_dir = []
        image_dir = 'Flowers_102/test_images'
        sem_seg_dir = 'Flowers_102/test_segmentations'
        val_data_dir = [image_dir, sem_seg_dir]
    
    elif cfg.GWM.DATASET in ["INTERNET"]:
        image_dir = 'rand_web_imgs'
        train_data_dir = [image_dir]
        val_data_dir = [image_dir]
        data_dirs = None

    else:
        raise ValueError('Unknown Setting/Dataset.')

    # Switching this section to pathlib, which should prevent double // errors in paths and dict keys
    root_path_str = cfg.GWM.DATA_ROOT    
    logger.info(f"Found DATA_ROOT in config: {root_path_str}")

    if root_path_str.startswith('/'):
        root_path = Path(f"/{root_path_str.lstrip('/').rstrip('/')}")
    else:
        root_path = Path(f"{root_path_str.lstrip('/').rstrip('/')}")

    logger.info(f"Loading dataset from: {root_path}")

    # basepath = root_path / basepath.lstrip('/').rstrip('/')
    # img_dir = root_path / img_dir.lstrip('/').rstrip('/')
    # pseudo_gt_dir = root_path / pseudo_gt_dir.lstrip('/').rstrip('/')
    # sem_seg_dir = root_path / sem_seg_dir.lstrip('/').rstrip('/')

    train_data_dir = [root_path / path.lstrip('/').rstrip('/') for path in train_data_dir]
    val_data_dir = [root_path / path.lstrip('/').rstrip('/') for path in val_data_dir]

    if data_dirs:
        for idx,dataset_dirs in enumerate(data_dirs):
            data_dirs[idx] = [root_path / path.lstrip('/').rstrip('/') for path in dataset_dirs]


    force1080p = ('DAVIS' not in cfg.GWM.DATASET) and 'RGB_BIG' in cfg.GWM.SAMPLE_KEYS

    enable_photometric_augmentations = cfg.FLAGS.INF_TPS

    # TODO. use the right dataset class (from datasets/custom_dataset.py) 
    train_dataset = DavisTrainDataset(data_dir=train_data_dir,
                                    resolution=resolution,
                                    size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY if not cfg.FLAGS.IGNORE_SIZE_DIV else -1     
                                    )
    
    if multi_val:  # False
        print(f"Using multiple validation datasets from {val_data_dir}")
        val_dataset = [FlowEvalDetectron(data_dir=val_data_dir,
                                         resolution=resolution,
                                         pair_list=pairs,
                                         val_seq=[vs],
                                         to_rgb=cfg.GWM.FLOW2RGB,
                                         with_rgb=False,
                                         size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY if not cfg.FLAGS.IGNORE_SIZE_DIV else -1,
                                         flow_clip=cfg.GWM.FLOW_CLIP,
                                         norm=cfg.GWM.FLOW_NORM,
                                         force1080p=force1080p) for vs in val_seq]
        for vs, vds in zip(val_seq, val_dataset):
            print(f"Validation dataset for {vs}: {len(vds)}")
            if len(vds) == 0:
                raise ValueError(f"Empty validation dataset for {vs}")

        if cfg.GWM.TTA_AS_TRAIN:
            if trainval_data_dir is None:
                trainval_data_dir = val_data_dir
            else:
                trainval_data_dir = [root_path / path.lstrip('/').rstrip('/') for path in trainval_data_dir]
            trainval_dataset = []
            tvd_basepath = root_path / str(trainval_data_dir[0].relative_to(root_path)).split('/')[0]
            print("TVD BASE DIR", tvd_basepath)
            for vs in val_seq:
                tvd_data_dir = [scan_train_flow([vs], res, pairs, tvd_basepath), *trainval_data_dir[1:]]
                tvd = FlowPairDetectron(data_dir=tvd_data_dir,
                                        resolution=resolution,
                                        to_rgb=cfg.GWM.FLOW2RGB,
                                        size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY if not cfg.FLAGS.IGNORE_SIZE_DIV else -1,
                                        enable_photo_aug=cfg.GWM.LOSS_MULT.EQV is not None,
                                        flow_clip=cfg.GWM.FLOW_CLIP,
                                        norm=cfg.GWM.FLOW_NORM,
                                        force1080p=force1080p,
                                        flow_res=cfg.GWM.FLOW_RES, )
                trainval_dataset.append(tvd)
                print(f'Seq {trainval_data_dir[0]}/{vs} dataset: {len(tvd)}')
        else:
            if trainval_data_dir is None:
                trainval_dataset = val_dataset
            else:
                trainval_data_dir = [root_path / path.lstrip('/').rstrip('/') for path in trainval_data_dir]
                trainval_dataset = []
                for vs in val_seq:
                    tvd = FlowEvalDetectron(data_dir=trainval_data_dir,
                                            resolution=resolution,
                                            pair_list=pairs,
                                            val_seq=[vs],
                                            to_rgb=cfg.GWM.FLOW2RGB,
                                            with_rgb=False,
                                            size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY if not cfg.FLAGS.IGNORE_SIZE_DIV else -1,
                                            flow_clip=cfg.GWM.FLOW_CLIP,
                                            norm=cfg.GWM.FLOW_NORM,
                                            force1080p=force1080p)
                    trainval_dataset.append(tvd)
                    print(f'Seq {trainval_data_dir[0]}/{vs} dataset: {len(tvd)}')
        return train_dataset, val_dataset, trainval_dataset
    
    # TODO use the correct validation dataset here.
    val_dataset = DavisValDataset(data_dir=val_data_dir,
                                resolution=resolution,
                                size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY if not cfg.FLAGS.IGNORE_SIZE_DIV else -1     
                            )

    return train_dataset, val_dataset


def loaders(cfg):
    train_dataset, val_dataset = setup_dataset(cfg)
    # logger.info(f"Sourcing data from {train_dataset.data_dir[0]}")

    if cfg.FLAGS.DEV_DATA:
        subset = cfg.SOLVER.IMS_PER_BATCH * 3
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(subset)))
        # val_dataset = torch.utils.data.Subset(val_dataset, list(range(subset)))

    g = torch.Generator()
    data_generator_seed = int(torch.randint(int(1e6), (1,)).item())
    logger.info(f"Dataloaders generator seed {data_generator_seed}")
    g.manual_seed(data_generator_seed)

    # train_loader = None
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               batch_size=cfg.SOLVER.IMS_PER_BATCH,   
                                               collate_fn=lambda x: x,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True,
                                               persistent_workers=cfg.DATALOADER.NUM_WORKERS > 0,
                                               worker_init_fn=utils.random_state.worker_init_function,
                                               generator=g
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=lambda x: x,
                                             drop_last=False,
                                             persistent_workers=cfg.DATALOADER.NUM_WORKERS > 0,
                                             worker_init_fn=utils.random_state.worker_init_function,
                                             generator=g)
    return train_loader, val_loader


def multi_loaders(cfg):
    train_dataset, val_datasets, train_val_datasets = setup_dataset(cfg, multi_val=True)
    logger.info(f"Sourcing multiple loaders from {len(val_datasets)}")
    logger.info(f"Sourcing data from {val_datasets[0].data_dir[0]}")

    g = torch.Generator()
    data_generator_seed = int(torch.randint(int(1e6), (1,)).item())
    logger.info(f"Dataloaders generator seed {data_generator_seed}")
    g.manual_seed(data_generator_seed)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                               collate_fn=lambda x: x,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True,
                                               persistent_workers=cfg.DATALOADER.NUM_WORKERS > 0,
                                               worker_init_fn=utils.random_state.worker_init_function,
                                               generator=g
                                               )

    val_loaders = [(torch.utils.data.DataLoader(val_dataset,
                                                num_workers=0,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=lambda x: x,
                                                drop_last=False,
                                                persistent_workers=False,
                                                worker_init_fn=utils.random_state.worker_init_function,
                                                generator=g),
                    torch.utils.data.DataLoader(tv_dataset,
                                                num_workers=0,
                                                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                                shuffle=True,
                                                pin_memory=False,
                                                collate_fn=lambda x: x,
                                                drop_last=False,
                                                persistent_workers=False,
                                                worker_init_fn=utils.random_state.worker_init_function,
                                                generator=g))
                   for val_dataset, tv_dataset in zip(val_datasets, train_val_datasets)]

    return train_loader, val_loaders


def add_gwm_config(cfg):
    cfg.GWM = CN()
    cfg.GWM.MODEL = "MASKFORMER"
    cfg.GWM.RESOLUTION = (256, 512)            
    cfg.GWM.FLOW_RES = (480, 854)
    cfg.GWM.SAMPLE_KEYS = ["image"]    
    cfg.GWM.ADD_POS_EMB = False
    cfg.GWM.CRITERION = "L2"
    cfg.GWM.L1_OPTIMIZE = False
    cfg.GWM.HOMOGRAPHY = 'quad'  # False
    cfg.GWM.HOMOGRAPHY_SUBSAMPLE = 8
    cfg.GWM.HOMOGRAPHY_SKIP = 0.4
    cfg.GWM.DATASET = 'DAVIS'
    cfg.GWM.DATA_ROOT = '../../'
    cfg.GWM.FLOW2RGB = False
    cfg.GWM.SIMPLE_REC = False
    cfg.GWM.DAVIS_SINGLE_VID = None
    cfg.GWM.USE_MULT_FLOW = False
    cfg.GWM.FLOW_COLORSPACE_REC = None

    cfg.GWM.FLOW_CLIP_U_LOW = float('-inf')
    cfg.GWM.FLOW_CLIP_U_HIGH = float('inf')
    cfg.GWM.FLOW_CLIP_V_LOW = float('-inf')
    cfg.GWM.FLOW_CLIP_V_HIGH = float('inf')

    cfg.GWM.FLOW_CLIP = float('inf')
    cfg.GWM.FLOW_NORM = False

    cfg.GWM.LOSS_MULT = CN()
    cfg.GWM.LOSS_MULT.REC = 0.03
    cfg.GWM.LOSS_MULT.HEIR_W = [0.1, 0.3, 0.6]

    cfg.GWM.TTA = 100  # Test-time-adaptation
    cfg.GWM.TTA_AS_TRAIN = False  # Use train-like data logic for test-time-adaptation

    cfg.GWM.LOSS = 'OG'

    cfg.FLAGS = CN()
    cfg.FLAGS.MAKE_VIS_VIDEOS = False  # Making videos is kinda slow
    cfg.FLAGS.EXTENDED_FLOW_RECON_VIS = False  # Does not cost much
    cfg.FLAGS.COMP_NLL_FOR_GT = False  # Should we log loss against ground truth?
    cfg.FLAGS.DEV_DATA = False
    cfg.FLAGS.KEEP_ALL = True  # Keep all checkoints
    cfg.FLAGS.ORACLE_CHECK = False  # Use oracle check to estimate max performance when grouping multiple components

    cfg.FLAGS.INF_TPS = False

    # cfg.FLAGS.UNFREEZE_AT = [(1, 10000), (0, 20000), (-1, 30000)]
    cfg.FLAGS.UNFREEZE_AT = [(4, 0), (2, 500), (1, 1000), (-1, 10000)]

    cfg.FLAGS.IGNORE_SIZE_DIV = False

    cfg.FLAGS.IGNORE_TMP = True

    cfg.WANDB = CN()
    cfg.WANDB.ENABLE = False
    cfg.WANDB.BASEDIR = '../'

    cfg.DEBUG = False

    cfg.LOG_ID = 'exp'
    cfg.LOG_FREQ = 5000   # 250
    cfg.OUTPUT_BASEDIR = '../outputs'
    cfg.SLURM = False
    cfg.SKIP_TB = False
    cfg.TOTAL_ITER = 25000
    cfg.CONFIG_FILE = './configs/maskformer/maskformer_R50_bs16_160k_dino.yaml'    # None

    if os.environ.get('SLURM_JOB_ID', None):
        cfg.LOG_ID = os.environ.get('SLURM_JOB_NAME', cfg.LOG_ID)
        logger.info(f"Setting name {cfg.LOG_ID} based on SLURM job name")

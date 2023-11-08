CUDA_VISIBLE_DEVICES=7 python main.py GWM.DATASET DAVIS LOG_ID davis # GWM.DATASET STv2 LOG_ID stv2
# [TODO] set train loader's batchSize=8 before running this script.
# torchrun main.py GWM.DATASET DAVIS LOG_ID davis
# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py GWM.DATASET STv2 LOG_ID stv2


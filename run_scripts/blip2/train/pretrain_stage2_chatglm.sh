CUDA_VISIBLE_DEVICES="6"  python -m torch.distributed.run --master_port 29600 --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_chatglm.yaml
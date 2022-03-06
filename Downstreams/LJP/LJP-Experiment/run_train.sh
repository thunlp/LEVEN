python -m torch.distributed.launch --nproc_per_node=4 train.py --config ./config/ljp/small/bert4fewshot.config --gpu 4,5,6,7 --do_test

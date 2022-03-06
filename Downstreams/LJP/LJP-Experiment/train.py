import argparse
from config_parser import create_config
import os
import torch
import logging

from tools.init_tool import init_all
from tools.train_tool import train

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--local_rank', default=-1, type=int, help='index of current process for distributed training')
    parser.add_argument('--do_test', help="do test while training or not", action="store_true")
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    args = parser.parse_args()

    config = create_config(args.config)
    config.set('distributed', 'local_rank', args.local_rank)

    use_gpu = args.gpu is not None
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        gpu_list = list(range(0, len(args.gpu.split(','))))

    os.system("clear")

    if config.getboolean("distributed", "use"):
        torch.cuda.set_device(gpu_list[args.local_rank])
        torch.distributed.init_process_group(backend=config.get("distributed", "backend"))
        config.set('distributed', 'gpu_num', len(gpu_list))

    logger.info("CUDA available: %s" % str(torch.cuda.is_available()))
    if not torch.cuda.is_available() and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    parameters = init_all(config, gpu_list, args.checkpoint, args.seed, "train", local_rank=args.local_rank)

    train(parameters, config, gpu_list, args.do_test, local_rank=args.local_rank)

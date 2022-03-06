import logging
import torch
import torch.nn as nn
import random
import numpy as np

from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
from .output_init import init_output_function

logger = logging.getLogger(__name__)


def init_all(config, gpu_list, checkpoint, seed, mode, *args, **params):
    result = {}

    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if len(gpu_list) > 0:
        torch.cuda.manual_seed_all(seed)

    logger.info("Begin to initialize dataset and formatter...")
    if mode == "train":
        init_formatter(config, ["train", "valid"], *args, **params)
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, *args, **params)
    else:
        init_formatter(config, ["test"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)

    logger.info("Begin to initialize models...")

    model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)
    optimizer = init_optimizer(model, config, *args, **params)
    trained_epoch = -1
    global_step = 0

    if len(gpu_list) > 0:
        if params['local_rank'] < 0:
            model = model.cuda()
        else:
            model = model.to(gpu_list[params['local_rank']])
        try:
            model.cuda(params['local_rank'])
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(params['local_rank'])
            model = nn.parallel.DistributedDataParallel(model, device_ids=[params['local_rank']])
        except Exception as e:
            logger.warning("No init_multi_gpu implemented in the model, use single gpu instead.")

    try:
        parameters = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(parameters["model"])

        if mode == "train":
            trained_epoch = parameters["trained_epoch"]
            if config.get("train", "optimizer") == parameters["optimizer_name"]:
                optimizer.load_state_dict(parameters["optimizer"])
            else:
                logger.warning("Optimizer changed, do not load parameters of optimizer.")

            if "global_step" in parameters:
                global_step = parameters["global_step"]
    except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        if mode == "test":
            logger.error(information)
            raise e
        else:
            logger.warning(information)

    result["model"] = model
    if mode == "train":
        result["optimizer"] = optimizer
        result["trained_epoch"] = trained_epoch
        result["output_function"] = init_output_function(config)
        result["global_step"] = global_step

    logger.info("Initialize done.")

    return result

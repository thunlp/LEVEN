import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
# from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import torch.distributed as dist
import json

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)


def valid(model, dataset, epoch, config, gpu_list, output_function, mode="valid"):
    model.eval()
    local_rank = config.getint('distributed', 'local_rank')

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = config.getint("output", "output_time")
    step = -1
    more = ""
    if total_len < 10000:
        more = "\t"

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        with torch.no_grad():
            results = model(data, config, gpu_list, acc_result, "valid")

        loss, acc_result = results["loss"], results["acc_result"]
        total_loss += float(loss)
        cnt += 1

        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    def gather(tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt.tolist()

    if config.getboolean("distributed", "use"):
        torch.distributed.barrier()

        # gather the acc_result in different processes
        for task in acc_result:
            if task in ['zm', 'ft']:
                for i in range(len(acc_result[task])):
                    for key in acc_result[task][i]:
                        acc_result[task][i][key] = gather(torch.tensor(acc_result[task][i][key]).cuda(local_rank))
            else:
                for i in range(len(acc_result[task])):
                    acc_result[task][i] = gather(torch.tensor(acc_result[task][i]).cuda(local_rank))

        output_info, output_dict = output_function(acc_result, config)

        if local_rank == 0:
            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    if local_rank < 0:
        delta_t = timer() - start_time
        output_info, _ = output_function(acc_result, config)
        output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                     "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    model.train()

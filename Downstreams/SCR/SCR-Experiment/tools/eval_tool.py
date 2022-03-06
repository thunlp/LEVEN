import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from timeit import default_timer as timer
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

    res_scores = []
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

        res_scores += list(zip(results["index"], results["score"]))
        if step % output_time == 0 and local_rank <= 0:
            delta_t = timer() - start_time

            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)
    del data
    del results

    predictions = {}
    for res in res_scores:
        print(res[1])
        if res[0][0] not in predictions:
            predictions[res[0][0]] = []
        predictions[res[0][0]].append((res[0][1], res[1]))

    for key in predictions:
        predictions[key].sort(key = lambda x:x[1], reverse = True)
        predictions[key] = [int(res[0]) for res in predictions[key]]

    os.makedirs(config.get("data", "result_path"), exist_ok=True)
    fout = open(os.path.join(config.get("data", "result_path"), "%s-test-%d_epoch-%d.json" % (config.get("output", "model_name"), config.getint("data", "test_file"), epoch)), "w")
    print(json.dumps(predictions), file = fout)
    fout.close()

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    if config.getboolean("distributed", "use"):
        shape = (len(acc_result), 4)

        mytensor = torch.LongTensor([[key["TP"], key["FN"], key["FP"], key["TN"]] for key in acc_result]).to(gpu_list[local_rank])
        mylist = [torch.LongTensor(shape[0], shape[1]).to(gpu_list[local_rank]) for i in range(config.getint('distributed', 'gpu_num'))]

        torch.distributed.all_gather(mylist, mytensor)#, 0)
        if local_rank == 0:
            mytensor = sum(mylist)
            index = 0
            for i in range(len(acc_result)):
                acc_result[i]['TP'], acc_result[i]['FN'], acc_result[i]['FP'], acc_result[i]['TN'] = int(mytensor[i][0]), int(mytensor[i][1]), int(mytensor[i][2]), int(mytensor[i][3])

    if local_rank <= 0:
        delta_t = timer() - start_time
        output_info = output_function(acc_result, config)
        output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    model.train()

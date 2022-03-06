import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.encoder.CNNEncoder import CNNEncoder
from model.loss import MultiLabelSoftmaxLoss, log_square_loss
from model.ljp.Predictor import LJPPredictor
from tools.accuracy_tool import multi_label_accuracy, log_distance_accuracy_function


class LJPCNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LJPCNN, self).__init__()

        self.encoder = CNNEncoder(config, gpu_list, *args, **params)
        self.fc = LJPPredictor(config, gpu_list, *args, **params)

        self.embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))),
                                      config.getint("model", "hidden_size"))

        self.criterion = {
            "zm": MultiLabelSoftmaxLoss(config, 202),
            "ft": MultiLabelSoftmaxLoss(config, 183),
            "xq": log_square_loss
        }
        self.accuracy_function = {
            "zm": multi_label_accuracy,
            "ft": multi_label_accuracy,
            "xq": log_distance_accuracy_function,
        }

    def init_multi_gpu(self, device, config, *args, **params):
        return

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']
        x = self.embedding(x)
        y = self.encoder(x)
        result = self.fc(y)

        loss = 0
        for name in ["zm", "ft", "xq"]:
            loss += self.criterion[name](result[name], data[name])

        if acc_result is None:
            acc_result = {"zm": None, "ft": None, "xq": None}

        for name in ["zm", "ft", "xq"]:
            acc_result[name] = self.accuracy_function[name](result[name], data[name], config, acc_result[name])

        return {"loss": loss, "acc_result": acc_result}

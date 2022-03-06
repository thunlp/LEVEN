import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from model.encoder.CNNEncoder import CNNEncoder
from model.loss import MultiLabelSoftmaxLoss, log_square_loss
from model.ljp.Predictor import LJPPredictor
from tools.accuracy_tool import multi_label_accuracy, log_distance_accuracy_function


class LSTMDecoder(nn.Module):
    def __init__(self, config):
        super(LSTMDecoder, self).__init__()
        self.feature_len = config.getint("model", "hidden_size")

        features = self.feature_len
        self.hidden_dim = features

        self.task_name = ["ft", "zm", "xq"]

        self.midfc = []
        for x in self.task_name:
            self.midfc.append(nn.Linear(features, features))

        self.cell_list = [None]
        for x in self.task_name:
            self.cell_list.append(nn.LSTMCell(self.feature_len, self.feature_len))

        self.hidden_state_fc_list = []
        for a in range(0, len(self.task_name) + 1):
            arr = []
            for b in range(0, len(self.task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []

        for a in range(0, len(self.task_name) + 1):
            arr = []
            for b in range(0, len(self.task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)

    def init_hidden(self, bs):
        self.hidden_list = []
        for a in range(0, len(self.task_name) + 1):
            self.hidden_list.append((torch.autograd.Variable(torch.zeros(bs, self.hidden_dim).cuda()),
                                     torch.autograd.Variable(torch.zeros(bs, self.hidden_dim).cuda())))

    def forward(self, x):
        fc_input = x
        outputs = {}
        batch_size = x.size()[0]
        self.init_hidden(batch_size)

        first = []
        for a in range(0, len(self.task_name) + 1):
            first.append(True)
        for a in range(1, len(self.task_name) + 1):
            h, c = self.cell_list[a](fc_input, self.hidden_list[a])
            for b in range(1, len(self.task_name) + 1):
                hp, cp = self.hidden_list[b]
                if first[b]:
                    first[b] = False
                    hp, cp = h, c
                else:
                    hp = hp + self.hidden_state_fc_list[a][b](h)
                    cp = cp + self.cell_state_fc_list[a][b](c)
                self.hidden_list[b] = (hp, cp)
            outputs[self.task_name[a - 1]] = self.midfc[a - 1](h).view(batch_size, -1)

        return outputs


class TopJudge(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(TopJudge, self).__init__()

        self.encoder = CNNEncoder(config, gpu_list, *args, **params)
        self.decoder = LSTMDecoder(config)

        self.fc = LJPPredictor(config, gpu_list, *args, **params)
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))

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

        self.embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))),
                                      config.getint("model", "hidden_size"))

    def init_multi_gpu(self, device, config, *args, **params):
        return
        self.encoder = nn.DataParallel(self.encoder, device_ids=device)
        self.decoder = nn.DataParallel(self.decoder, device_ids=device)
        self.dropout = nn.DataParallel(self.dropout, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']
        x = self.embedding(x)
        hidden = self.encoder(x)
        hidden = self.dropout(hidden)
        result = self.decoder(hidden)
        for name in result:
            result[name] = self.fc(self.dropout(result[name]))[name]

        loss = 0
        for name in ["zm", "ft", "xq"]:
            loss += self.criterion[name](result[name], data[name])

        if acc_result is None:
            acc_result = {"zm": None, "ft": None, "xq": None}

        for name in ["zm", "ft", "xq"]:
            acc_result[name] = self.accuracy_function[name](result[name], data[name], config, acc_result[name])

        return {"loss": loss, "acc_result": acc_result}

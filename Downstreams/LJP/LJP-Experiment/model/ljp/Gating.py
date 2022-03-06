import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
from collections import OrderedDict

from model.encoder.LSTMEncoder import LSTMEncoder
from model.encoder.CNNEncoder import CNNEncoder
from model.loss import MultiLabelSoftmaxLoss, log_square_loss
from model.ljp.Predictor import LJPPredictor
from tools.accuracy_tool import multi_label_accuracy, log_distance_accuracy_function

cx = None


class GatingLayer(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(GatingLayer, self).__init__()

        num_layers = config.getint("model", "num_layers")
        config.set("model", "num_layers", 1)
        self.encoder = LSTMEncoder(config, gpu_list, *args, **params)
        config.set("model", "num_layers", num_layers)

        self.hidden_size = config.getint("model", "hidden_size")
        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))

    def init_multi_gpu(self, device, config, *args, **params):
        return

    def forward(self, h):
        _, h = self.encoder(h)
        g = self.fc(torch.cat([h, cx], dim=2))
        h = g * h
        h = self.dropout(h)
        return h


class Gating(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Gating, self).__init__()

        self.encoder = []
        num_layers = config.getint("model", "num_layers")
        for a in range(0, num_layers):
            self.encoder.append(("Layer%d" % a, GatingLayer(config, gpu_list, *args, **params)))

        self.encoder = nn.Sequential(OrderedDict(self.encoder))
        self.fc = LJPPredictor(config, gpu_list, *args, **params)

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))), self.hidden_size)
        self.charge_embedding = nn.Embedding(202, self.hidden_size)

        self.fake_tensor = []
        for a in range(0, 202):
            self.fake_tensor.append(a)
        self.fake_tensor = Variable(torch.LongTensor(self.fake_tensor)).cuda()
        self.cnn = CNNEncoder(config, gpu_list, *args, **params)
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

    def init_multi_gpu(self, device, config, *args, **params):
        return

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['text']
        x = self.word_embedding(x)

        batch_size = x.size()[0]
        c = self.fake_tensor.view(1, -1).repeat(batch_size, 1)
        c = self.charge_embedding(c)
        zm = data["zm"].view(batch_size, -1, 1).repeat(1, 1, self.hidden_size)
        c = c * zm.float()
        c = torch.max(c, dim=1)[0]

        global cx
        cx = c.view(batch_size, 1, -1).repeat(1, config.getint("data", "max_seq_length"), 1)
        y = self.encoder(x)
        y = self.dropout(y)
        y = self.cnn(y)
        y = self.dropout(y)

        result = self.fc(y)

        loss = 0
        for name in ["zm", "ft", "xq"]:
            loss += self.criterion[name](result[name], data[name])

        if acc_result is None:
            acc_result = {"zm": None, "ft": None, "xq": None}

        for name in ["zm", "ft", "xq"]:
            acc_result[name] = self.accuracy_function[name](result[name], data[name], config, acc_result[name])

        return {"loss": loss, "acc_result": acc_result}

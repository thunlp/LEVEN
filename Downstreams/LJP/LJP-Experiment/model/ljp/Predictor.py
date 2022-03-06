import torch
import torch.nn as nn
import torch.nn.functional as F


class LJPPredictor(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LJPPredictor, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.charge_fc = nn.Linear(self.hidden_size, 202 * 2)
        self.article_fc = nn.Linear(self.hidden_size, 183 * 2)
        self.term_fc = nn.Linear(self.hidden_size, 1)

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, h):
        charge = self.charge_fc(h)
        article = self.article_fc(h)
        term = self.term_fc(h)

        batch = h.size()[0]
        charge = charge.view(batch, -1, 2)
        article = article.view(batch, -1, 2)
        term = term.view(batch)

        return {"zm": charge, "ft": article, "xq": term}

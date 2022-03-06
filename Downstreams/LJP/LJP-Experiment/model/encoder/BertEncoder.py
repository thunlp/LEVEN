import torch.nn as nn
from transformers import BertModel
from model.encoder.BertEncoder_with_event import EventBertModel


class BertEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertEncoder, self).__init__()

        if config.getboolean('model', 'use_event') or config.getboolean('model', 'use_event_type'):
            print('using EDBERT (Event Detection BERT)')
            print('use event: ', config.getboolean('model', 'use_event'))
            print('use event type: ', config.getboolean('model', 'use_event_type'))
            model = EventBertModel.from_pretrained(config.get('model', 'bert_path'))
        else:
            print('using original BERT')
            model = BertModel.from_pretrained(config.get("model", "bert_path"))

        self.bert = model

    def forward(self, x):
        outputs = self.bert(**x)
        return outputs['pooler_output']

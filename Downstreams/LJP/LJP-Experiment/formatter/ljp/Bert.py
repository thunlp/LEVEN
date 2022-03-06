import torch
import numpy as np

from transformers import BertTokenizer
from formatter.Basic import BasicFormatter


class BertLJP(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

        charge = open(config.get("data", "charge_path"), "r", encoding="utf8")
        self.charge2id = {}

        for line in charge:
            self.charge2id[line.replace("\r", "").replace("\n", "")] = len(self.charge2id)

        article = open(config.get("data", "article_path"), "r", encoding="utf8")
        self.article2id = {}

        for line in article:
            self.article2id[int(line.replace("\r", "").replace("\n", ""))] = len(self.article2id)

    def process(self, data, config, mode, *args, **params):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        event_type_ids = []
        charge = []
        article = []
        term = []

        for temp in data:
            # inputs
            use_event = config.getboolean('model', 'use_event')
            use_event_type = config.getboolean('model', 'use_event_type')

            if use_event or use_event_type:
                input_ids.append(temp['inputs']['input_ids'])
                attention_mask.append(temp['inputs']['attention_mask'])

                if use_event:
                    token_type_ids.append(temp['inputs']['token_type_ids'])

                if use_event_type:
                    event_type_ids.append(temp['inputs']['event_type_ids'])

            else:
                text = temp["fact"]
                text = self.tokenizer.tokenize(text)
                while len(text) < self.max_len:
                    text.append("[PAD]")
                text = text[0:self.max_len]
                input_ids.append(self.tokenizer.convert_tokens_to_ids(text))

            # charge
            temp_charge = np.zeros(len(self.charge2id), dtype=np.int)
            for name in temp["meta"]["accusation"]:
                temp_charge[self.charge2id[name.replace("[", "").replace("]", "")]] = 1
            charge.append(temp_charge.tolist())

            # article
            temp_article = np.zeros(len(self.article2id), dtype=np.int)
            for name in temp["meta"]["relevant_articles"]:
                temp_article[self.article2id[int(name)]] = 1
            article.append(temp_article.tolist())

            # term
            if temp["meta"]["term_of_imprisonment"]["life_imprisonment"]:
                temp_term = 350
            elif temp["meta"]["term_of_imprisonment"]["death_penalty"]:
                temp_term = 400
            else:
                temp_term = int(temp["meta"]["term_of_imprisonment"]["imprisonment"])
            term.append(temp_term)

        if config.getboolean('model', 'use_event'):
            inputs = {'input_ids': input_ids,
                      'token_type_ids': token_type_ids,
                      'attention_mask': attention_mask,
                      'event_type_ids': event_type_ids}
        else:
            inputs = {'input_ids': input_ids}

        for key in inputs:
            if inputs[key]:
                inputs[key] = torch.LongTensor(inputs[key])
            else:
                inputs[key] = None

        charge = torch.LongTensor(charge)
        article = torch.LongTensor(article)
        term = torch.FloatTensor(term)

        return {'text': inputs, 'zm': charge, 'ft': article, 'xq': term}

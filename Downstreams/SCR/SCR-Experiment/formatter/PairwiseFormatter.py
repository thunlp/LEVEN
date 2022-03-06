import json
import torch
import os
import numpy as np

from transformers import AutoTokenizer, BertTokenizer
from formatter.Basic import BasicFormatter


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class PairwiseFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))
        self.mode = mode
        self.query_len = config.getint("train", "query_len")
        self.cand_len = config.getint("train", "cand_len")
        self.max_len = self.query_len + self.cand_len + 3
        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.cls_id = self.tokenizer.cls_token_id

    def process(self, data, config, mode, *args, **params):
        use_event = config.getboolean('model', 'use_event')

        inputx = []
        segment = []
        mask = []
        if use_event:
            event = []
        labels = []

        for pairs in data:
            inputx.append([])
            segment.append([])
            mask.append([])
            if use_event:
                event.append([])

            for temp in pairs:
                if use_event:
                    query_input_ids = temp['query_inputs']['input_ids']
                    cand_input_ids = temp['cand_inputs']['input_ids']

                    input_ids = [self.cls_id] + query_input_ids + [self.sep_id] + cand_input_ids + [self.sep_id]
                    segment_ids = [0] * (len(query_input_ids) + 2) + [1] * (len(cand_input_ids) + 1)
                    input_mask = [1] * len(input_ids)

                    query_event_ids = temp['query_inputs']['event_type_ids']
                    cand_event_ids = temp['cand_inputs']['event_type_ids']
                    event_ids = [0] + query_event_ids + [0] + cand_event_ids + [0]

                else:
                    query = self.tokenizer.tokenize(temp["query"])[:self.query_len]
                    cand = self.tokenizer.tokenize(temp["cand"])[:self.cand_len]
                    tokens = ["[CLS]"] + query + ["[SEP]"] + cand + ["[SEP]"]

                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    segment_ids = [0] * (len(query) + 2) + [1] * (len(cand) + 1)
                    input_mask = [1] * len(input_ids)

                padding = [0] * (self.max_len - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                if use_event:
                    event_ids += padding

                assert len(input_ids) == self.max_len
                assert len(input_mask) == self.max_len
                assert len(segment_ids) == self.max_len
                if use_event:
                    if len(event_ids) != self.max_len:
                        print(len(event_ids))
                    assert len(event_ids) == self.max_len

                inputx[-1].append(input_ids)
                segment[-1].append(segment_ids)
                mask[-1].append(input_mask)
                if use_event:
                    event[-1].append(event_ids)

            # if mode != "train" or pairs[0]["label"] == pairs[1]["label"]:
            #     labels.append(-100)
            # else:
            #     labels.append(int(pairs[1]["label"]))
            labels.append(int(pairs[0]['label']))

        if mode == "train":
            global_att = np.zeros((len(data), 2, self.max_len), dtype=np.int32)
        else:
            global_att = np.zeros((len(data), 1, self.max_len), dtype=np.int32)
        global_att[:, :, 0] = 1

        ret = {
            "inputx": torch.LongTensor(inputx),
            "segment": torch.LongTensor(segment),
            "mask": torch.LongTensor(mask),
            "event": torch.LongTensor(event) if use_event else None,
            "global_att": torch.LongTensor(global_att),
            "labels": torch.LongTensor(labels),
        }
        if mode != "train":
            ret["index"] = [temp[0]["index"] for temp in data]
        return ret

import enum
import json
import random
import torch
import os
import numpy as np

from transformers import AutoTokenizer,BertTokenizer
from formatter.Basic import BasicFormatter
import joblib
import jieba
from tqdm import tqdm


class RecurrentFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))
        self.mode = mode

        self.block_len = config.getint("train", "block_len")
        self.max_block_size = config.getint("train", "block_num")

        self.punctuations = set([511, 8043, 8013, 8039]) # 包括 ，。？！  这里改了的话一定记得去改模型里面给sentence id的部分
    
    def split_sents(self, doc):
        sents = []
        last_pos = 1
        for tpos, token in enumerate(doc[1:-1]):
            if token in self.punctuations:
                sents.append(doc[last_pos:tpos + 2])
                last_pos = tpos + 2
        sents.append(doc[last_pos:-1])
        return [s for s in sents if len(s) > 0]

    def split_for_recurrent(self, query, cand, block_len=512, pad_id=0, max_block_size=8):
        blocks, now_inp_block = [], []
        mask = []
        for doc in [query, cand]:
            doc = self.tokenizer.encode(doc)
            now_inp_block.append(doc[0])
            sents = self.split_sents(doc)
            for i in range(len(sents)):
                if len(blocks) >= max_block_size:
                    break
                if len(sents[i]) > block_len:
                    sents[i] = sents[i][:block_len]
                if block_len - len(now_inp_block) < len(sents[i]):
                    mask.append(([1] * len(now_inp_block) + [0] * (block_len - len(now_inp_block)))[:block_len])
                    now_inp_block += [pad_id] * (block_len - len(now_inp_block))
                    blocks.append(now_inp_block[:block_len])
                    now_inp_block = []
                now_inp_block += sents[i]
            if len(now_inp_block) > 0:
                now_inp_block.append(doc[-1])
        if len(now_inp_block) > 0 and len(blocks) < max_block_size:
            mask.append(([1] * len(now_inp_block) + [0] * (block_len - len(now_inp_block)))[:block_len])
            now_inp_block += [pad_id] * (block_len - len(now_inp_block))
            blocks.append(now_inp_block[:block_len])

        for b in blocks:
            assert len(b) == block_len
        if len(blocks) < max_block_size:
            mask += [[0] * block_len] * (max_block_size - len(blocks))
            blocks += [[pad_id] * block_len] * (max_block_size - len(blocks))
        assert len(blocks) == max_block_size

        return blocks, mask

    def process(self, data, config, mode, *args, **params):
        inpblocks = []
        mask = []
        labels = []
        for temp in data:
            blocks, bmask = self.split_for_recurrent(temp["query"], temp["cand"], self.block_len, self.tokenizer.pad_token_id, self.max_block_size)

            inpblocks.append(blocks)
            mask.append(bmask)
            labels.append(int(temp["label"]))
            # print(blocks)
        return {
            "mask": torch.LongTensor(mask),
            "inp": torch.LongTensor(inpblocks),
            "labels": torch.LongTensor(labels),
            "index": [temp["index"] for temp in data]
        }

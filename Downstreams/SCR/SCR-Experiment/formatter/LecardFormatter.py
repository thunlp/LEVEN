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

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    # print(len(tokens_a), len(tokens_b))
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class LecardFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = AutoTokenizer.from_pretrained(config.get('train', 'PLM_vocab'))
        self.mode = mode
        self.query_len = config.getint("train", "query_len")
        self.cand_len = config.getint("train", "cand_len")
        self.max_len = self.query_len + self.cand_len + 3
        self.gat_strategy = config.get("train", "gat_strategy")
        if self.gat_strategy == "QueryAtt":
            self.get_gat = self.QueryAtt
        elif self.gat_strategy == "RandAtt":
            self.get_gat = self.RandAtt
        elif self.gat_strategy == "PeriodAtt":
            self.get_gat = self.PeriodAtt
        elif self.gat_strategy == "TfidfAtt":
            self.get_gat = self.TfidfAtt
            tfidf = joblib.load("/home/xcj/LegalLongPLM/lecard/tfidf.joblib")
            self.idfs = {v: tfidf.idf_[tfidf.vocabulary_[v]] for v in tqdm(tfidf.vocabulary_.keys())}
    
    def QueryAtt(self, query, input_ids):
        ret = [1] * (len(query) + 2)
        ret += [0] * (len(input_ids) - len(ret))
        return ret
    
    def RandAtt(self, query, input_ids):
        poses = list(range(len(input_ids)))
        selected = set(random.sample(poses, len(query)))
        selected.add(0)
        return [1 if i in selected else 0 for i in range(len(input_ids))]
    
    def PeriodAtt(self, query, input_ids):
        ret = [1 if token == 511 or token == 8024 else 0 for token in input_ids]
        ret[0] = 1
        return ret
    '''
    def TfidfAtt(self, query, input_ids, qtext, ctext):
        gat_mask = np.zeros(len(input_ids))
        qtokens = list(jieba.cut(qtext))
        ctokens = list(jieba.cut(ctext))
        qtfidf = {}
        for token in qtokens:
            if token not in qtfidf:
                qtfidf[token] = 0
            qtfidf[token] += 1
        ctfidf = {}
        for token in ctokens:
            if token not in ctfidf:
                ctfidf[token] = 0
            ctfidf[token] += 1
        for token in qtfidf:
            if token not in self.idfs:
                qtfidf[token] = 0
            else:
                qtfidf[token] = qtfidf[token] * self.idfs[token]
        for token in ctfidf:
            if token not in self.idfs:
                ctfidf[token] = 0
            else:
                ctfidf[token] = ctfidf[token] * self.idfs[token]
        qsort = sorted(qtfidf.items(), key=lambda x:x[1], reverse=True)
        csort = sorted(ctfidf.items(), key=lambda x:x[1], reverse=True)
        qgtoken = {token[0] for token in qsort[:int(len(query) * 0.3)]}
        cgtoken = {token[0] for token in csort[:len(query) - len(qgtoken)]}
        qtokenbeg = [0] + np.cumsum([len(token) for token in qtokens]).tolist()
        ctokenbeg = [0] + np.cumsum([len(token) for token in ctokens]).tolist()
        for tid, token in enumerate(qtokens):
            if qtokenbeg[tid + 1] > self.query_len:
                continue
            if token in qgtoken:
                print(qtokenbeg[tid] + 1, qtokenbeg[tid + 1] + 1, end=", ")
                gat_mask[qtokenbeg[tid] + 1: qtokenbeg[tid + 1] + 1] = 1
        for tid, token in enumerate(ctokens):
            if ctokenbeg[tid + 1] > self.cand_len:
                continue
            if token in cgtoken:
                print(ctokenbeg[tid] + len(query) + 2, ctokenbeg[tid + 1] + len(query) + 2, end=",")
                gat_mask[ctokenbeg[tid] + len(query) + 2: ctokenbeg[tid + 1] + len(query) + 2] = 1
        print("==" * 20)
        print("good token in query", qgtoken)
        print("good token in cand", cgtoken)
        print("query", qtext)
        print("candidate", ctext)
        print(gat_mask.tolist())
        print(self.tokenizer.convert_ids_to_tokens([input_ids[i] for i in range(len(input_ids)) if gat_mask[i] == 1]))
        return gat_mask.tolist()
    '''
    def cal_tfidf(self, qtokens, ctokens):
        qtfidf = {}
        for token in qtokens:
            if token not in qtfidf:
                qtfidf[token] = 0
            qtfidf[token] += 1
        ctfidf = {}
        for token in ctokens:
            if token not in ctfidf:
                ctfidf[token] = 0
            ctfidf[token] += 1
        for token in qtfidf:
            if token not in self.idfs:
                qtfidf[token] = 0
            else:
                qtfidf[token] = qtfidf[token] * self.idfs[token]
        for token in ctfidf:
            if token not in self.idfs:
                ctfidf[token] = 0
            else:
                ctfidf[token] = ctfidf[token] * self.idfs[token]
        return qtfidf, ctfidf

    def TfidfAtt(self, qtext, ctext):
        qtokens = list(jieba.cut(qtext))
        ctokens = list(jieba.cut(ctext))
        
        qtfidf, ctfidf = self.cal_tfidf(qtokens, ctokens)

        qids = [] # self.tokenizer.tokenize(token) for token in qtokens]
        qgatprob = []
        for token in qtokens:
            tids = self.tokenizer.tokenize(token)
            qids += tids
            qgatprob += [qtfidf[token]] * len(tids)
        qids, qgatprob = qids[:self.query_len], qgatprob[:self.query_len]

        cids = [] # self.tokenizer.tokenize(token) for token in ctokens]
        cgatprob = []
        for token in ctokens:
            tids = self.tokenizer.tokenize(token)
            cids += tids
            cgatprob += [ctfidf[token]] * len(tids)
        cids, cgatprob = cids[:self.cand_len], cgatprob[:self.cand_len]

        input_ids = ["[CLS]"] + qids + ["[SEP]"] + cids + ["[SEP]"]
        qgatprob, cgatprob = np.array(qgatprob), np.array(cgatprob)
        qpos = np.random.choice(list(range(len(qids))), size = int(len(qids) * 0.4), p=qgatprob/qgatprob.sum())
        cpos = np.random.choice(list(range(len(cids))), size = len(qids) - len(qpos), p=cgatprob/cgatprob.sum())

        qgat = np.zeros(len(qids))
        qgat[qpos] = 1
        cgat = np.zeros(len(cids))
        cgat[cpos] = 1
        gat = [1] + qgat.tolist() + [1] + cgat.tolist() + [1]
        
        segment_ids = [0] * (len(qids) + 2) + [1] * (len(cids) + 1)

        # print("==" * 20)
        # print("query", qtext)
        # print("candidate", ctext)
        # print([input_ids[i] for i in range(len(input_ids)) if gat[i] == 1])
        return self.tokenizer.convert_tokens_to_ids(input_ids), gat, segment_ids

    def process(self, data, config, mode, *args, **params):
        inputx = []
        segment = []
        mask = []
        labels = []
        global_att = []
        for temp in data:
            query = self.tokenizer.tokenize(temp["query"])[:self.query_len]
            cand = self.tokenizer.tokenize(temp["cand"])[:self.cand_len]

            tokens = ["[CLS]"] + query + ["[SEP]"] + cand + ["[SEP]"]
            segment_ids = [0] * (len(query) + 2) + [1] * (len(cand) + 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if self.gat_strategy == "TfidfAtt":
                input_ids, gat_mask, segment_ids = self.get_gat(temp["query"], temp["cand"])
            else:
                gat_mask = self.get_gat(query, input_ids)
            input_mask = [1] * len(input_ids)
            # gat_mask = [1] * (len(query) + 2)

            padding = [0] * (self.max_len - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            gat_mask += padding
            # gat_mask += [0] * (self.max_len - len(gat_mask))

            assert len(input_ids) == self.max_len
            assert len(input_mask) == self.max_len
            assert len(segment_ids) == self.max_len
            assert len(gat_mask) == self.max_len

            inputx.append(input_ids)
            segment.append(segment_ids)
            mask.append(input_mask)
            labels.append(int(temp["label"]))
            global_att.append(gat_mask)

        #global_att = np.zeros((len(data), self.max_len), dtype=np.int32)
        #global_att[:,0] = 1
        return {
            "inputx": torch.LongTensor(inputx),
            "segment": torch.LongTensor(segment),
            "mask": torch.LongTensor(mask),
            "global_att": torch.LongTensor(global_att),
            "labels": torch.LongTensor(labels),
            "index": [temp["index"] for temp in data]
        }

#!/usr/bin/env python
import sys
import os
import os.path
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    truth_file = os.path.join(truth_dir, "test_gold.jsonl")
    truth = open(truth_file, "r", encoding='utf-8')

    submission_answer_file = os.path.join(submit_dir, "results.jsonl")
    submission_answer = open(submission_answer_file, "r")
    preds_map = dict()
    ans_lines = submission_answer.readlines()
    for line in ans_lines:
        data = json.loads(line)
        tmp = dict()
        for mention in data['predictions']:
            tmp[mention['id']] = mention['type_id']
        preds_map[data['id']] = tmp

    ref_lines = truth.readlines()
    labels = []
    preds = []
    for line in ref_lines:
        data = json.loads(line)
        pred_tmp = preds_map[data['id']] if data['id'] in preds_map else dict()
        if not pred_tmp:  # debug
            print("lose", data['id'])
        for event in data['events']:
            for mention in event['mention']:
                if mention['id'] in pred_tmp:
                    preds.append(pred_tmp[mention['id']])
                else:
                    preds.append(0)
                    print("lose Mention", mention['id'])
                labels.append(event['type_id'])
        for mention in data['negative_triggers']:
            if mention['id'] in pred_tmp:
                preds.append(pred_tmp[mention['id']])
            else:
                preds.append(0)
                print("lose Mention", mention['id'])
            labels.append(0)
    assert len(labels) == len(preds)

    # calculate scores
    pos_labels = list(range(1, 109))
    labels = np.array(labels)
    preds = np.array(preds)
    micro_p = precision_score(labels, preds, labels=pos_labels, average='micro') * 100.0
    micro_r = recall_score(labels, preds, labels=pos_labels, average='micro') * 100.0
    micro_f1 = f1_score(labels, preds, labels=pos_labels, average='micro') * 100.0

    macro_p = precision_score(labels, preds, labels=pos_labels, average='macro') * 100.0
    macro_r = recall_score(labels, preds, labels=pos_labels, average='macro') * 100.0
    macro_f1 = f1_score(labels, preds, labels=pos_labels, average='macro') * 100.0

    print("Micro_F1:", micro_f1)
    print("Micro_Precision:", micro_p)
    print("Micro_Recall:", micro_r)
    print("Macro_F1:", macro_f1)
    print("Macro_Precision:", macro_p)
    print("Macro_Recall:", macro_r)

    with open(os.path.join(output_dir, 'scores.txt'), 'w') as f:
        scores = {'Micro_F1': micro_f1, 'Micro_Precision': micro_p, 'Micro_Recall': micro_r,
                  'Macro_F1': macro_f1, 'Macro_Precision': macro_p, 'Macro_Recall': macro_r}
        json.dump(scores, f, indent=4, ensure_ascii=False)

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT-CRF fine-tuning: utilities to work with LEVEN. """

from __future__ import absolute_import, division, print_function
import logging
import os
import jsonlines
from transformers import XLMRobertaTokenizer, BertTokenizer, RobertaTokenizer

from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

bio_labels = [
    "O",
    "B-明知",
    "I-明知",
    "B-投案",
    "I-投案",
    "B-供述",
    "I-供述",
    "B-谅解",
    "I-谅解",
    "B-赔偿",
    "I-赔偿",
    "B-退赃",
    "I-退赃",
    "B-销赃",
    "I-销赃",
    "B-分赃",
    "I-分赃",
    "B-搜查/扣押",
    "I-搜查/扣押",
    "B-举报",
    "I-举报",
    "B-拘捕",
    "I-拘捕",
    "B-报警/报案",
    "I-报警/报案",
    "B-鉴定",
    "I-鉴定",
    "B-冲突",
    "I-冲突",
    "B-言语冲突",
    "I-言语冲突",
    "B-肢体冲突",
    "I-肢体冲突",
    "B-买卖",
    "I-买卖",
    "B-卖出",
    "I-卖出",
    "B-买入",
    "I-买入",
    "B-租/借",
    "I-租/借",
    "B-出租/出借",
    "I-出租/出借",
    "B-租用/借用",
    "I-租用/借用",
    "B-归还/偿还",
    "I-归还/偿还",
    "B-获利",
    "I-获利",
    "B-雇佣",
    "I-雇佣",
    "B-放贷",
    "I-放贷",
    "B-集资",
    "I-集资",
    "B-支付/给付",
    "I-支付/给付",
    "B-签订合同/订立协议",
    "I-签订合同/订立协议",
    "B-制造",
    "I-制造",
    "B-遗弃",
    "I-遗弃",
    "B-运输/运送",
    "I-运输/运送",
    "B-邮寄",
    "I-邮寄",
    "B-组织/安排",
    "I-组织/安排",
    "B-散布",
    "I-散布",
    "B-联络",
    "I-联络",
    "B-通知/提醒",
    "I-通知/提醒",
    "B-介绍/引荐",
    "I-介绍/引荐",
    "B-邀请/招揽",
    "I-邀请/招揽",
    "B-纠集",
    "I-纠集",
    "B-阻止/妨碍",
    "I-阻止/妨碍",
    "B-挑衅/挑拨",
    "I-挑衅/挑拨",
    "B-帮助/救助",
    "I-帮助/救助",
    "B-提供",
    "I-提供",
    "B-放纵",
    "I-放纵",
    "B-跟踪",
    "I-跟踪",
    "B-同意/接受",
    "I-同意/接受",
    "B-拒绝/抗拒",
    "I-拒绝/抗拒",
    "B-放弃/停止",
    "I-放弃/停止",
    "B-要求/请求",
    "I-要求/请求",
    "B-建议",
    "I-建议",
    "B-约定",
    "I-约定",
    "B-饮酒",
    "I-饮酒",
    "B-自然灾害",
    "I-自然灾害",
    "B-洪涝",
    "I-洪涝",
    "B-干旱",
    "I-干旱",
    "B-山体滑坡",
    "I-山体滑坡",
    "B-事故",
    "I-事故",
    "B-交通事故",
    "I-交通事故",
    "B-火灾事故",
    "I-火灾事故",
    "B-爆炸事故",
    "I-爆炸事故",
    "B-暴力",
    "I-暴力",
    "B-杀害",
    "I-杀害",
    "B-伤害人身",
    "I-伤害人身",
    "B-言语辱骂",
    "I-言语辱骂",
    "B-敲诈勒索",
    "I-敲诈勒索",
    "B-威胁/强迫",
    "I-威胁/强迫",
    "B-持械/持枪",
    "I-持械/持枪",
    "B-拘束/拘禁",
    "I-拘束/拘禁",
    "B-绑架",
    "I-绑架",
    "B-欺骗",
    "I-欺骗",
    "B-拐骗",
    "I-拐骗",
    "B-冒充",
    "I-冒充",
    "B-伪造",
    "I-伪造",
    "B-变造",
    "I-变造",
    "B-盗窃财物",
    "I-盗窃财物",
    "B-抢夺财物",
    "I-抢夺财物",
    "B-抢劫财物",
    "I-抢劫财物",
    "B-挪用财物",
    "I-挪用财物",
    "B-侵占财物",
    "I-侵占财物",
    "B-毁坏财物",
    "I-毁坏财物",
    "B-猥亵",
    "I-猥亵",
    "B-强奸",
    "I-强奸",
    "B-卖淫",
    "I-卖淫",
    "B-嫖娼",
    "I-嫖娼",
    "B-吸毒",
    "I-吸毒",
    "B-贩卖毒品",
    "I-贩卖毒品",
    "B-赌博",
    "I-赌博",
    "B-开设赌场",
    "I-开设赌场",
    "B-指使/教唆",
    "I-指使/教唆",
    "B-共谋",
    "I-共谋",
    "B-违章驾驶",
    "I-违章驾驶",
    "B-泄露信息",
    "I-泄露信息",
    "B-私藏/藏匿",
    "I-私藏/藏匿",
    "B-入室/入户",
    "I-入室/入户",
    "B-贿赂",
    "I-贿赂",
    "B-逃匿",
    "I-逃匿",
    "B-放火",
    "I-放火",
    "B-走私",
    "I-走私",
    "B-投毒",
    "I-投毒",
    "B-自杀",
    "I-自杀",
    "B-死亡",
    "I-死亡",
    "B-受伤",
    "I-受伤",
    "B-被困",
    "I-被困",
    "B-中毒",
    "I-中毒",
    "B-昏迷",
    "I-昏迷",
    "B-遗失",
    "I-遗失",
    "B-受损",
    "I-受损"
]


class InputExample(object):
    """
        A single training/test example for token classification.
        one single sequence of tokens is an example in LEVEN task.
    """

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    input_data = jsonlines.open(os.path.join(data_dir, "{}.jsonl".format(mode)))
    examples = []

    for doc in input_data:
        words = [c['tokens'] for c in doc['content']]
        labels = [['O']*len(c['tokens']) for c in doc['content']]

        if mode != 'test':
            for event in doc['events']:
                for mention in event['mention']:
                    labels[mention['sent_id']][mention['offset'][0]] = "B-" + event['type']
                    for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                        labels[mention['sent_id']][i] = "I-" + event['type']

            for mention in doc['negative_triggers']:
                labels[mention['sent_id']][mention['offset'][0]] = "O"
                for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                    labels[mention['sent_id']][i] = "O"

        for i in range(0, len(words)):
            examples.append(InputExample(guid="%s-%s-%d" % (mode, doc['id'], i),
                                         words=words[i],
                                         labels=labels[i]))

    return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-100,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 model_name=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    # my logic in crf_padding requires this check. I create mask for crf by labels==pad_token_label_id to not include it
    # in loss and decoding
    assert pad_token_label_id not in label_map.values()

    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            print("###############")
            logger.info("Writing example %d of %d", ex_index, len(examples))
            print("###############")

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                word_tokens = ['<UNK>']
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        '''
        The convention in BERT is:
        (a) For sequence pairs:
         tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
         type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        (b) For single sequences:
         tokens:   [CLS] the dog is hairy . [SEP]
         type_ids:   0   0   0   0  0     0   0

        Where "type_ids" are used to indicate whether this is the first
        sequence or the second sequence. The embedding vectors for `type=0` and
        `type=1` were learned during pre-training and are added to the wordpiece
        embedding vector (and position vector). This is not *strictly* necessary
        since the [SEP] token unambiguously separates the sequences, but it makes
        it easier for the model to learn the concept of sequences.
        '''

        tokens += [sep_token]
        label_ids += [pad_token_label_id]       # [label_map["X"]]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        if model_name:
            if model_name == 'xlm-roberta-base':
                tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
            elif model_name.startswith('bert'):
                tokenizer = BertTokenizer.from_pretrained(model_name)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
            elif model_name == 'roberta':
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids

        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", "".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", "".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", "".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", "".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", "".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids))
    return features


def get_labels():
    return bio_labels


def to_crf_pad(org_array, org_mask, pad_label_id):
    crf_array = [aa[bb] for aa, bb in zip(org_array, org_mask)]
    crf_array = pad_sequence(crf_array, batch_first=True, padding_value=pad_label_id)
    crf_pad = (crf_array != pad_label_id)
    # the viterbi decoder function in CRF makes use of multiplicative property of 0, then pads wrong numbers out.
    # Need a*0 = 0 for CRF to work.
    crf_array[~crf_pad] = 0
    return crf_array, crf_pad


def unpad_crf(returned_array, returned_mask, org_array, org_mask):
    out_array = org_array.clone().detach()
    out_array[org_mask] = returned_array[returned_mask]
    return out_array

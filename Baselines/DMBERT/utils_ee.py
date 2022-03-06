# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 Xiaozhi Wang
# Copyright 2021 Feng Yao
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
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """
import json
import logging
import os
from typing import List
import jsonlines
from tqdm import tqdm
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)
event2id = {
    "None": 0,
    "明知": 1,
    "投案": 2,
    "供述": 3,
    "谅解": 4,
    "赔偿": 5,
    "退赃": 6,
    "销赃": 7,
    "分赃": 8,
    "搜查/扣押": 9,
    "举报": 10,
    "拘捕": 11,
    "报警/报案": 12,
    "鉴定": 13,
    "冲突": 14,
    "言语冲突": 15,
    "肢体冲突": 16,
    "买卖": 17,
    "卖出": 18,
    "买入": 19,
    "租/借": 20,
    "出租/出借": 21,
    "租用/借用": 22,
    "归还/偿还": 23,
    "获利": 24,
    "雇佣": 25,
    "放贷": 26,
    "集资": 27,
    "支付/给付": 28,
    "签订合同/订立协议": 29,
    "制造": 30,
    "遗弃": 31,
    "运输/运送": 32,
    "邮寄": 33,
    "组织/安排": 34,
    "散布": 35,
    "联络": 36,
    "通知/提醒": 37,
    "介绍/引荐": 38,
    "邀请/招揽": 39,
    "纠集": 40,
    "阻止/妨碍": 41,
    "挑衅/挑拨": 42,
    "帮助/救助": 43,
    "提供": 44,
    "放纵": 45,
    "跟踪": 46,
    "同意/接受": 47,
    "拒绝/抗拒": 48,
    "放弃/停止": 49,
    "要求/请求": 50,
    "建议": 51,
    "约定": 52,
    "饮酒": 53,
    "自然灾害": 54,
    "洪涝": 55,
    "干旱": 56,
    "山体滑坡": 57,
    "事故": 58,
    "交通事故": 59,
    "火灾事故": 60,
    "爆炸事故": 61,
    "暴力": 62,
    "杀害": 63,
    "伤害人身": 64,
    "言语辱骂": 65,
    "敲诈勒索": 66,
    "威胁/强迫": 67,
    "持械/持枪": 68,
    "拘束/拘禁": 69,
    "绑架": 70,
    "欺骗": 71,
    "拐骗": 72,
    "冒充": 73,
    "伪造": 74,
    "变造": 75,
    "盗窃财物": 76,
    "抢夺财物": 77,
    "抢劫财物": 78,
    "挪用财物": 79,
    "侵占财物": 80,
    "毁坏财物": 81,
    "猥亵": 82,
    "强奸": 83,
    "卖淫": 84,
    "嫖娼": 85,
    "吸毒": 86,
    "贩卖毒品": 87,
    "赌博": 88,
    "开设赌场": 89,
    "指使/教唆": 90,
    "共谋": 91,
    "违章驾驶": 92,
    "泄露信息": 93,
    "私藏/藏匿": 94,
    "入室/入户": 95,
    "贿赂": 96,
    "逃匿": 97,
    "放火": 98,
    "走私": 99,
    "投毒": 100,
    "自杀": 101,
    "死亡": 102,
    "受伤": 103,
    "被困": 104,
    "中毒": 105,
    "昏迷": 106,
    "遗失": 107,
    "受损": 108
}

class InputExample(object):
    """A single training/test example"""
    def __init__(self, example_id, tokens, triggerL, triggerR, label=None):
        self.example_id = example_id
        self.tokens = tokens
        self.triggerL = triggerL
        self.triggerR = triggerR
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, input_ids, input_mask, segment_ids, maskL, maskR, label):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.maskL = maskL
        self.maskR = maskR
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class LEVENProcessor(DataProcessor):
    """Processor for the LEVEN data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(os.path.join(data_dir, 'train.jsonl'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(os.path.join(data_dir, 'valid.jsonl'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(os.path.join(data_dir, 'test.jsonl'), "test")

    def get_labels(self):
        """See base class."""
        return list(event2id.keys())

    @staticmethod
    def _create_examples(fin, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        input_data = jsonlines.open(fin)

        for data in input_data:
            for event in data['events']:
                for mention in event['mention']:
                    e_id = "%s-%s" % (set_type, mention['id'])
                    examples.append(
                        InputExample(
                            example_id=e_id,
                            tokens=data['content'][mention['sent_id']]['tokens'],
                            triggerL=mention['offset'][0],
                            triggerR=mention['offset'][1],
                            label=event['type'],
                        )
                    )
            for nt in data['negative_triggers']:
                e_id = "%s-%s" % (set_type, nt['id'])
                examples.append(
                    InputExample(
                        example_id=e_id,
                        tokens=data['content'][nt['sent_id']]['tokens'],
                        triggerL=nt['offset'][0],
                        triggerR=nt['offset'][1],
                        label='None',
                    )
                )

        return examples


class LEVENInferProcessor(DataProcessor):
    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(os.path.join(data_dir, 'test.jsonl'), "test")

    def get_labels(self):
        """See base class."""

        return list(event2id.keys())

    @staticmethod
    def _create_examples(fin, set_type):
        """Creates examples for the test sets."""
        examples = []
        input_data = jsonlines.open(fin)
        for data in input_data:
            for mention in data['candidates']:
                e_id = "%s-%s" % (set_type, mention['id'])
                examples.append(
                    InputExample(
                        example_id=e_id,
                        tokens=data['content'][mention['sent_id']]['tokens'],
                        triggerL=mention['offset'][0],
                        triggerR=mention['offset'][1],
                        label='None',
                    )
                )
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(tqdm(examples, desc='convert examples to features')):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # leven is in Chinese, therefore, use "".join() instead of " ".join()
        textL = tokenizer.tokenize("".join(example.tokens[:example.triggerL]))

        textR = tokenizer.tokenize("".join(example.tokens[example.triggerL:example.triggerR]))
        textR += ['[unused1]']
        textR += tokenizer.tokenize("".join(example.tokens[example.triggerR:]))

        maskL = [1.0 for i in range(0, len(textL)+1)] + [0.0 for i in range(0, len(textR)+2)]
        maskR = [0.0 for i in range(0, len(textL)+1)] + [1.0 for i in range(0, len(textR)+2)]

        if len(maskL) > max_length:
            maskL = maskL[:max_length]
        if len(maskR) > max_length:
            maskR = maskR[:max_length]

        inputs = tokenizer.encode_plus(textL + ['[unused0]'] + textR,
                                       add_special_tokens=True,
                                       max_length=max_length,
                                       return_token_type_ids=True,
                                       return_overflowing_tokens=True)

        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! you are cropping tokens."
            )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        assert len(input_ids) == len(maskL)
        assert len(input_ids) == len(maskR)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            maskL = ([0.0] * padding_length) + maskL
            maskR = ([0.0] * padding_length) + maskR
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            maskL = maskL + ([0.0] * padding_length)
            maskR = maskR + ([0.0] * padding_length)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        label = label_map[example.label]

        if ex_index < 0:    # dont print
            logger.info("*** Example ***")
            logger.info("example_id: {}".format(example.example_id))
            logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
            logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
            logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
            logger.info("maskL: {}".format(" ".join(map(str, maskL))))
            logger.info("maskR: {}".format(" ".join(map(str, maskR))))
            logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id,
                                      input_ids=input_ids,
                                      input_mask=attention_mask,
                                      segment_ids=token_type_ids,
                                      maskL=maskL,
                                      maskR=maskR,
                                      label=label))

    return features


processors = {"leven": LEVENProcessor, "leven_infer": LEVENInferProcessor}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"leven", 109}

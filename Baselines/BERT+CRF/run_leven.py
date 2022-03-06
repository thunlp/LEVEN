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
""" BERT-CRF running for LEVEN """

import argparse
import glob
import os
import random
import json

import numpy as np
import seqeval.metrics as se

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    DistilBertConfig,
    RobertaConfig,
    XLMRobertaConfig,
    get_linear_schedule_with_warmup,
)
from utils_leven import convert_examples_to_features, get_labels, read_examples_from_file
from bert_crf import *

import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "bertcrf": (BertConfig, BertCRFForTokenClassification, BertTokenizer),
}

pure_event2id = {
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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    best_dev_f1 = 0.0
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    # for crf might wanna change this to SGD with gradient clipping and my other scheduler
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    best_steps = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]

            outputs = model(pad_token_label_id=pad_token_label_id, **inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="valid")
                        if results['f1-micro'] > best_dev_f1:
                            best_dev_f1 = results['f1-micro']
                            best_steps = global_step
                            # results_test, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id,
                            #                            mode="test")
                            # logger.info(
                            #     "test f1: %s, loss: %s",
                            #     str(results_test['f1']),
                            #     str(results_test['loss']),
                            # )
                            if best_steps:
                                logger.info("best steps of eval f1 is the following checkpoints: %s", best_steps)
                                logger.info("best precision: {}, recall: {}, f1: {}".format(results['p-micro'],
                                                                                            results['r-micro'],
                                                                                            results['f1-micro']))
                                with open('./saved/best_step_info.txt', 'w', encoding='utf-8') as f:
                                    json.dump({'best_checkpoint': best_steps, 'f1': results['eval_f1'],
                                               'precision': results['eval_p'], 'recall': results['eval_recall']}, f,
                                              indent=4, ensure_ascii=False)

                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model

                    logger.info("Saving model checkpoint to %s", output_dir)

                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))

                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(pad_token_label_id=pad_token_label_id, **inputs)
            tmp_eval_loss, logits, best_path = outputs

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        if preds is None:
            preds = best_path.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, best_path.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        # "loss": eval_loss,
        "p-micro": se.precision_score(out_label_list, preds_list, average='micro'),
        "r-micro": se.recall_score(out_label_list, preds_list, average='micro'),
        "f1-micro": se.f1_score(out_label_list, preds_list, average='micro'),

        "p-marco": se.precision_score(out_label_list, preds_list, average='macro'),
        "r-marco": se.recall_score(out_label_list, preds_list, average='macro'),
        "f1-macro": se.f1_score(out_label_list, preds_list, average='macro'),
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in results.keys():
        logger.info("  %s = %s", key, str(results[key]))

    info = se.classification_report(out_label_list, preds_list, digits=4, output_dict=True)
    for key in info:
        value = info[key]
        for v in value:
            value[v] = float(value[v])
    with open('./saved/prf_report_dict.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    print(se.classification_report(out_label_list, preds_list, digits=4), file=open('./saved/prf_report.txt', 'w'))

    return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir, 
        "cached_{}_{}_{}".format(mode,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length))
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples, 
            labels, 
            args.max_seq_length, tokenizer,
            cls_token_at_end=False,           # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,            # roberta uses an extra separator b/w pairs of sentences
            pad_on_left=False,                # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            pad_token_label_id=pad_token_label_id
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    use_parser = True
    parser = argparse.ArgumentParser()
    args = None

    if use_parser:
        # Required parameters
        parser.add_argument("--data_dir", default='./data', type=str, required=False,
                            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")

        parser.add_argument("--model_type", default=None, type=str, required=True,
                            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

        parser.add_argument("--model_name_or_path", default='./bert-base-chinese', type=str, required=False,
                            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                                ALL_MODELS))

        parser.add_argument("--output_dir", default='./saved', type=str, required=False,
                            help="The output directory where the model predictions and checkpoints will be written.")

        # Other parameters
        parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--cache_dir", default="", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--max_seq_length", default=512, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                 "than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--do_train", action="store_true",
                            help="Whether to run training.")
        parser.add_argument("--do_eval", action="store_true",
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--do_infer", action="store_true",
                            help="Whether to run inference on the test set.")
        parser.add_argument("--evaluate_during_training", action="store_true",
                            help="Whether to run evaluation during training at each logging step.")
        parser.add_argument("--do_lower_case", action="store_true",
                            help="Set this flag if you are using an uncased model.")

        parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--learning_rate", default=5e-5, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float,
                            help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--num_train_epochs", default=1.0, type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="Linear warmup over warmup_steps.")

        parser.add_argument("--logging_steps", type=int, default=1000,
                            help="Log every X updates steps.")
        parser.add_argument("--save_steps", type=int, default=500,
                            help="Save checkpoint every X updates steps.")
        parser.add_argument("--eval_all_checkpoints", action="store_true",
                            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
        parser.add_argument("--no_cuda", action="store_true",
                            help="Avoid using CUDA when available")
        parser.add_argument("--overwrite_output_dir", action="store_true",
                            help="Overwrite the content of the output directory")
        parser.add_argument("--overwrite_cache", action="store_true",
                            help="Overwrite the cached training and evaluation sets")
        parser.add_argument("--seed", type=int, default=42,
                            help="random seed for initialization")

        parser.add_argument("--fp16", action="store_true",
                            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
        parser.add_argument("--fp16_opt_level", type=str, default="O1",
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
        parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
        args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        logger.info({"n_gpu: ": args.n_gpu})

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARN)

    # Set seed
    set_seed(args)

    labels = get_labels()
    num_labels = len(labels)

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    # pad_token_label_id = CrossEntropyLoss().ignore_index
    pad_token_label_id = -100

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    ner_model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    ner_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, train_dataset, ner_model, tokenizer, labels, pad_token_label_id)
        print(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            ner_model.module if hasattr(ner_model, "module") else ner_model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    best_dev_f1 = 0.0
    best_steps = 0
    best_checkpoint = None
    if args.do_eval and args.local_rank in [-1, 0]:
        print('####################################################')
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            print('')
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="valid", prefix=global_step)

            if result['f1-micro'] > best_dev_f1:
                best_dev_f1 = result['f1-micro']
                best_steps = global_step
                if best_steps:
                    logger.info("best steps of eval f1 is the following checkpoints: %s", best_steps)
                    logger.info("best precision: {}, recall: {}, f1: {}".format(result['p-micro'],
                                                                                result['r-micro'],
                                                                                result['f1-micro']))
                    with open('./saved/best_step_info.txt', 'w', encoding='utf-8') as f:
                        json.dump({'best_checkpoint': best_steps,
                                   **result},
                                  f,
                                  indent=4,
                                  ensure_ascii=False)
                    best_checkpoint = checkpoint

            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        logger.info("best steps of eval f1 is the following checkpoints: %s", best_steps)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_infer and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "results.jsonl")
        with open(output_test_predictions_file, "w") as writer:
            Cnt = 0
            levenTypes = list(pure_event2id.keys())
            with open(os.path.join(args.data_dir, "test.jsonl"), "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    doc = json.loads(line)
                    res = {}
                    res['id'] = doc['id']
                    res['predictions'] = []
                    for mention in doc['candidates']:
                        if mention['offset'][1] > len(predictions[Cnt + mention['sent_id']]):
                            print(len(doc['content'][mention['sent_id']]['tokens']),
                                  len(predictions[Cnt + mention['sent_id']]))
                            res['predictions'].append({"id": mention['id'], "type_id": 0})
                            continue
                        is_NA = False if predictions[Cnt + mention['sent_id']][mention['offset'][0]].startswith(
                            "B") else True
                        if not is_NA:
                            Type = predictions[Cnt + mention['sent_id']][mention['offset'][0]][2:]
                            for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                                if predictions[Cnt + mention['sent_id']][i][2:] != Type:
                                    is_NA = True
                                    break
                            if not is_NA:
                                res['predictions'].append({"id": mention['id'], "type_id": levenTypes.index(Type)})
                        if is_NA:
                            res['predictions'].append({"id": mention['id'], "type_id": 0})
                    writer.write(json.dumps(res) + "\n")
                    Cnt += len(doc['content'])

    return results


if __name__ == "__main__":
    main()

from transformers import BertTokenizer
from bert_crf import BertCRFForTokenClassification
import torch
from tqdm import tqdm
import jsonlines
import os
import warnings
import json
import re
import random
import math
warnings.filterwarnings('ignore')


def split_sentence(text):
    sent_list = re.split(r'([。])', text) + [""]
    sent_list = ["".join(item) for item in zip(sent_list[0::2], sent_list[1::2])]
    sent_list = [sent for sent in sent_list if sent.strip()]

    return sent_list


def generate_sentence_batch(input_data, batch_size, mode='query'):
    sent2doc = {}
    all_sentences = []
    for doc_id, doc in enumerate(input_data):
        sent_list = split_sentence(doc['q'] if mode == 'query' else doc['ajjbqk'])

        for sent_id in range(len(all_sentences), len(all_sentences)+len(sent_list)):
            sent2doc[sent_id] = doc_id

        all_sentences += [{'doc_id': doc_id,
                           'sent_id': sid,
                           'sentence': sent} for sid, sent in zip(range(len(all_sentences),
                                                                        len(all_sentences)+len(sent_list)),
                                                                  sent_list)]

    all_sentences = sorted(all_sentences, key=lambda x: len(x['sentence']), reverse=True)

    output_batches = []
    for i in range(len(all_sentences) // batch_size + 1):
        sub = all_sentences[i * batch_size:(i + 1) * batch_size]
        if sub:
            output_batches.append(sub)

    return output_batches


def add_inputs(input_batches):
    for batch in input_batches:
        sentence_list = [b['sentence'] for b in batch]

        input_ids_list = tokenizer(sentence_list, add_special_tokens=False)['input_ids']

        # forward inputs
        inputs = tokenizer(sentence_list, padding='longest', truncation=True, return_tensors='pt')
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            bio_predictions = model(**inputs, pad_token_label_id=-100)[1]

            mask = (bio_predictions != -100).long().to(device)
            bio_predictions = bio_predictions * mask
            label_predictions = (bio_predictions * mask + 1) // 2

            for i, prediction in enumerate(label_predictions):
                # select the non-padding tokens and remove [CLS] and [SEP] at the beginning and the end
                event_ids = torch.masked_select(prediction, inputs['attention_mask'][i].bool()).tolist()[1:-1]

                # do padding if the sentence is truncated
                if len(event_ids) < len(input_ids_list[i]):
                    event_ids += [0] * (len(input_ids_list[i]) - len(event_ids))

                assert len(input_ids_list[i]) == len(event_ids)

                batch[i]['input_ids'] = input_ids_list[i]
                batch[i]['event_ids'] = event_ids

    return sorted(sum(input_batches, []), key=lambda x: (x['doc_id'], x['sent_id']))


def write_batch(batches, full_data, file_name):
    current_doc_id = 0
    inputs = {'input_ids': [],
              'event_ids': []}

    with jsonlines.open(file_name, 'w') as f:
        for i, d in enumerate(batches):
            if d['doc_id'] == current_doc_id and i != len(batches) - 1:
                inputs['input_ids'] += d['input_ids']
                inputs['event_ids'] += d['event_ids']
            else:
                if i == len(batches) - 1:
                    inputs['input_ids'] += d['input_ids']
                    inputs['event_ids'] += d['event_ids']

                jsonlines.Writer.write(f, {**full_data[current_doc_id], 'inputs': inputs})
                inputs = {'input_ids': [],
                          'event_ids': []}

                inputs['input_ids'] += d['input_ids']
                inputs['event_ids'] += d['event_ids']
                current_doc_id += 1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using {} device'.format(device))

    model = BertCRFForTokenClassification.from_pretrained('./saved/checkpoint-1900').to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    os.makedirs('./output_data-unsupervised/query', exist_ok=True)
    os.makedirs('./output_data-unsupervised/candidates', exist_ok=True)

    # process query data
    data_path = './input_data/query/query.json'
    save_path = './output_data-unsupervised/query/query.json'

    full_data = list(jsonlines.open(data_path))
    batches = generate_sentence_batch(full_data, batch_size=50)
    batches = add_inputs(batches)

    write_batch(batches, full_data=full_data, file_name=save_path)

    # process candidates data
    data_path = './input_data/candidates'
    save_path = './output_data-unsupervised/candidates'
    for folder in tqdm(os.listdir(data_path), desc='writing {}'.format('candidate')):
        os.makedirs(os.path.join(save_path, folder), exist_ok=True)

        full_data = []
        files = []
        for file in os.listdir(os.path.join(data_path, folder)):
            full_data.append({**json.load(open(os.path.join(data_path, folder, file), encoding='utf-8'))})
            files.append(file)

        batches = generate_sentence_batch(full_data, batch_size=50, mode='candidates')
        batches = add_inputs(batches)

        merged_batches = {}
        for b in batches:
            did = b['doc_id']
            if did not in merged_batches:
                merged_batches[did] = [b]
            else:
                merged_batches[did].append(b)

        for batch in merged_batches.values():
            did = batch[0]['doc_id']
            with jsonlines.open(os.path.join(save_path, folder, files[did]), 'w') as f:
                inputs = {'input_ids': [],
                          'event_ids': []}
                for b in batch:
                    inputs['input_ids'] += b['input_ids']
                    inputs['event_ids'] += b['event_ids']
                jsonlines.Writer.write(f, {**full_data[did], 'inputs': inputs})

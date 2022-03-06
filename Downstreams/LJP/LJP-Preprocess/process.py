from transformers import BertTokenizer
from bert_crf import BertCRFForTokenClassification
import torch
from tqdm import tqdm
import jsonlines
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class BertCrfTagging(object):
    def __init__(self,
                 data_path='./input_data/small',
                 save_path='./output_data/small',
                 checkpoint='./saved/checkpoint-1900',
                 max_length=512,
                 batch_size=16,
                 device=torch.device('cpu')):
        self.device = device
        self.model = BertCRFForTokenClassification.from_pretrained(checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.save_path = save_path
        print('using {} device'.format(device))

    def process(self):
        for file in os.listdir(self.data_path):
            mode = file[0:file.index('.')]
            data = list(jsonlines.open(os.path.join(self.data_path, file)))

            data = self.generate_batch(data)
            with jsonlines.open(os.path.join(self.save_path, file), 'w') as f:
                for i, batch in enumerate(tqdm(data, desc='writing {}.json'.format(mode))):
                    data[i] = self.add_inputs(batch)
                    for d in data[i]:
                        jsonlines.Writer.write(f, d)

    def generate_batch(self, input_data):
        batches = []
        for i in range(len(input_data)//self.batch_size+1):
            batches.append(input_data[i*self.batch_size:(i+1)*self.batch_size])

        return batches

    def add_inputs(self, batch):
        facts = [b['fact'] for b in batch]

        # dynamic padding
        # inputs = self.tokenizer.batch_encode_plus(facts, max_length=512)
        # max_length = min(max([len(ipt) for ipt in inputs['input_ids']]), 512)
        max_length = 512
        inputs = self.tokenizer.batch_encode_plus(facts,
                                                  max_length=max_length,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt')

        # shift tensors to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        # forward pass
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, pad_token_label_id=-100)
            bio_predictions = outputs[1]

            pad_mask = (bio_predictions != -100).long().to(self.device)
            bio_predictions = bio_predictions * pad_mask

            token_type_ids = (bio_predictions != 0).long()
            evt_predictions = (bio_predictions + 1) // 2

        inputs['token_type_ids'] = token_type_ids
        inputs['event_type_ids'] = evt_predictions

        for i, b in enumerate(batch):
            input_ids = inputs['input_ids'][i].tolist()
            token_type_ids = inputs['token_type_ids'][i].tolist()
            attention_mask = inputs['attention_mask'][i].tolist()
            event_type_ids = inputs['event_type_ids'][i].tolist()
            batch[i]['inputs'] = {'input_ids': input_ids,
                                  'token_type_ids': token_type_ids,
                                  'attention_mask': attention_mask,
                                  'event_type_ids': event_type_ids}
        return batch


if __name__ == "__main__":
    bert_tagging = BertCrfTagging(data_path='./input_data/small',
                                  save_path='./output_data/small',
                                  checkpoint='./saved/checkpoint-1900',
                                  batch_size=50,
                                  device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
    bert_tagging.process()

import logging
from transformers import BertModel, BertPreTrainedModel
from transformers import BertConfig

from crf import *
from utils_leven import to_crf_pad, unpad_crf

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin"
}


class BertCRFForTokenClassification(BertPreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(BertCRFForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels + 2)
        self.crf = CRF(self.num_labels)

        self.init_weights()

    def _get_features(self, input_ids=None, attention_mask=None, token_type_ids=None,
                      position_ids=None, head_mask=None, inputs_embeds=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        feats = self.classifier(sequence_output)
        return feats, outputs

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, pad_token_label_id=None):

        logits, outputs = self._get_features(input_ids, attention_mask, token_type_ids,
                                             position_ids, head_mask, inputs_embeds)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            pad_mask = (labels != pad_token_label_id)

            # Only keep active parts of the loss
            if attention_mask is not None:
                loss_mask = ((attention_mask == 1) & pad_mask)
            else:
                loss_mask = ((torch.ones(logits.shape) == 1) & pad_mask)

            crf_labels, crf_mask = to_crf_pad(labels, loss_mask, pad_token_label_id)
            crf_logits, _ = to_crf_pad(logits, loss_mask, pad_token_label_id)

            loss = self.crf.neg_log_likelihood(crf_logits, crf_mask, crf_labels)
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            best_path = self.crf(crf_logits, crf_mask)  # (torch.ones(logits.shape) == 1)
            best_path = unpad_crf(best_path, crf_mask, labels, pad_mask)
            outputs = (loss,) + outputs + (best_path,)
        else:
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            if attention_mask is not None:
                mask = (attention_mask == 1)  # & (labels!=-100))
            else:
                mask = torch.ones(logits.shape).bool()  # (labels!=-100)
            crf_logits, crf_mask = to_crf_pad(logits, mask, pad_token_label_id)
            crf_mask = crf_mask.sum(axis=2) == crf_mask.shape[2]
            best_path = self.crf(crf_logits, crf_mask)
            temp_labels = torch.ones(mask.shape) * pad_token_label_id
            best_path = unpad_crf(best_path, crf_mask, temp_labels, mask)
            outputs = outputs + (best_path,)

        return outputs

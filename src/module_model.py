# -*- coding: UTF-8 -*-

# @Date    : Sep 9, 2020
# @Author  : Nrothblue
"""模型及各个模块"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

import globalvar as gl
seed = gl.get_value('seed')
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)


bert_path = '../model/emb/bert-mini/'
dropout = 0.15

sent_hidden_size = 256
sent_num_layers = 2

class WhitespaceTokenizer():
    """WhitespaceTokenizer with vocab."""

    def __init__(self):
        vocab_file = bert_path + 'vocab.txt'
        self._token2id = self.load_vocab(vocab_file)
        self._id2token = {v: k for k, v in self._token2id.items()}
        self.max_len = 256
        self.unk = 1

        logging.info("Build Bert vocab with size %d." % (self.vocab_size))

    def load_vocab(self, vocab_file):
        f = open(vocab_file, 'r')
        lines = f.readlines()
        lines = list(map(lambda x: x.strip(), lines))
        vocab = dict(zip(lines, range(len(lines))))
        return vocab

    def tokenize(self, tokens):
        assert len(tokens) <= self.max_len - 2
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        output_tokens = self.token2id(tokens)
        return output_tokens

    def token2id(self, xs):
        if isinstance(xs, list):
            return [self._token2id.get(x, self.unk) for x in xs]
        return self._token2id.get(xs, self.unk)

    @property
    def vocab_size(self):
        return len(self._id2token)


class WordBertEncoder(nn.Module):
    """Word Encoder"""

    def __init__(self):
        super(WordBertEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.tokenizer = WhitespaceTokenizer()
        self.bert = BertModel.from_pretrained(bert_path)

        self.pooled = False
        logging.info('Build Bert encoder with pooled {}.'.format(self.pooled))

    def encode(self, tokens):
        tokens = self.tokenizer.tokenize(tokens)
        return tokens

    def get_bert_parameters(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def forward(self, input_ids, token_type_ids):
        # input_ids: sen_num x bert_len
        # token_type_ids: sen_num  x bert_len

        # sen_num x bert_len x 256, sen_num x 256
        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)

        if self.pooled:
            reps = pooled_output
        else:
            reps = sequence_output[:, 0, :]  # sen_num x 256

        if self.training:
            reps = self.dropout(reps)

        return reps

class SentEncoder(nn.Module):
    """Sent Encoder"""

    def __init__(self, sent_rep_size):
        super(SentEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.sent_lstm = nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens



class Attention(nn.Module):
    """Sent Attention"""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)

        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))

        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # linear
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # b x len x hidden

        # compute attention
        outputs = torch.matmul(key, self.query)  # b x len

        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))

        attn_scores = F.softmax(masked_outputs, dim=1)  # b x len

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # sum weighted sources
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden

        return batch_outputs, attn_scores


class Model(nn.Module):
    """Model Complete Flow"""

    def __init__(self, vocab):
        super(Model, self).__init__()
        self.sent_rep_size = 256
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordBertEncoder()
        bert_parameters = self.word_encoder.get_bert_parameters()

        self.sent_encoder = SentEncoder(self.sent_rep_size)
        self.sent_attention = Attention(self.doc_rep_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))

        self.out = nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        use_cuda = gl.get_value('use_cuda')
        device = gl.get_value('device')
        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters
        self.all_parameters["bert_parameters"] = bert_parameters

        logging.info('Build model with bert word encoder, lstm sent encoder.')

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len

        sent_reps = self.word_encoder(batch_inputs1, batch_inputs2)  # sen_num x sent_rep_size

        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)  # b x doc_len x sent_rep_size
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)  # b x doc_len x max_sent_len
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)  # b x doc_len x doc_rep_size
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)  # b x doc_rep_size

        batch_outputs = self.out(doc_reps)  # b x num_labels

        return batch_outputs

# -*- coding: UTF-8 -*-

# @Date    : Sep 9, 2020
# @Author  : Nrothblue
"""数据加载器"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

from tqdm import tqdm, tqdm_notebook
import numpy as np

import globalvar as gl
seed = gl.get_value('seed')
np.random.seed(seed)

def sentence_split(text, vocab, max_sent_len=256, max_segment=16):
    """文档切分成句子"""

    words = text.strip().split()
    document_len = len(words)

    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        segments.append([len(segment), segment])

    assert len(segments) > 0
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        return segments


def get_examples(data, word_encoder, vocab, max_sent_len=256, max_segment=8):
    """文档根据字典重新编码"""

    label2id = vocab.label2id
    examples = []

    for text, label in tqdm(zip(data['text'], data['label'])):
        # label
        id = label2id(label)

        # words
        sents_words = sentence_split(text, vocab, max_sent_len-2, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            token_ids = word_encoder.encode(sent_words)
            sent_len = len(token_ids)
            token_type_ids = [0] * sent_len
            doc.append([sent_len, token_ids, token_type_ids])
        examples.append([id, len(doc), doc])

    logging.info('Total %d docs.' % len(examples))
    return examples


def batch_slice(data, batch_size):
    """batch 切分"""

    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield docs


def data_iter(data, batch_size, shuffle=True, noise=1.0):
    """数据迭代器
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle:
        np.random.shuffle(data)

        lengths = [example[1] for example in data]
        noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
        sorted_indices = np.argsort(noisy_lengths).tolist()
        sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data

    batched_data.extend(list(batch_slice(sorted_data, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch

#!/usr/bin/env python
# encoding: utf-8
'''
@author: zessay
@license: (C) Copyright Sogou.
@contact: zessay@sogou-inc.com
@file: ner_span.py
@time: 2020/3/4 21:03
@description: 
'''
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor, InputExample, get_entities

logger = logging.getLogger(__name__)


class InputFeature(object):
    def __init__(self, input_ids, input_mask, input_len, segment_ids, start_ids, end_ids, subjects):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        self.input_len = input_len
        self.end_ids = end_ids
        self.subjects = subjects

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

# ---------------------- 用于数据封装之后的后处理函数 ----------------------------

def collate_fn(batch):
    all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    all_start_ids = all_start_ids[:,:max_len]
    all_end_ids = all_end_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens

def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
    label2id = {label:i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")
        text = example.text_a
        subjects = example.labels
        tokens = tokenizer.tokenize(text)
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)
        subjects_id = []
        for subject in subjects:
            label = subject[0]
            start = subject[1]
            end = subject[2]
            start_ids[start] = label2id[label]
            end_ids[end] = label2id[label]
            subjects_id.append((label2id[label], start, end))
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length-special_tokens_count)]
            start_ids = start_ids[: (max_seq_length-special_tokens_count)]
            end_ids = end_ids[: (max_seq_length-special_tokens_count)]

        tokens += [sep_token]
        start_ids += [0]
        end_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            start_ids += [0]
            end_ids += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            start_ids = [0] + start_ids
            end_ids = [0] + end_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            start_ids = ([0] * padding_length) + start_ids
            end_ids = ([0] * padding_length) + end_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            start_ids += ([0] * padding_length)
            end_ids += ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("start_ids: %s" % " ".join([str(x) for x in start_ids]))
            logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))

        features.append(InputFeature(input_ids=input_ids,
                                     input_mask=input_mask,
                                     segment_ids=segment_ids,
                                     start_ids=start_ids,
                                     end_ids=end_ids,
                                     subjects=subjects_id,
                                     input_len=input_len))
    return features

# ------------------------- 用于Span标记预测的CNER数据 --------------------------

class CNerProcessor(DataProcessor):
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            subject = get_entities(labels,id2label=None,markup='bios')
            examples.append(InputExample(guid=guid, text_a=text_a, subject=subject))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "CONT", "ORG","LOC",'EDU','NAME','PRO','RACE','TITLE']

# ------------------------- 用于Span标记预测的ClueNER数据 --------------------------

class ClueNerProcessor(DataProcessor):
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = line['labels']
            subject = get_entities(labels,id2label=None,markup='bios')
            examples.append(InputExample(guid=guid, text_a=text_a, labels=subject))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "address", "book","company",'game','government','movie','name','organization','position','scene']

# ----------------------------------------------------------------------------------

ner_processors = {
    "cner": CNerProcessor,
    "cluener": ClueNerProcessor
}

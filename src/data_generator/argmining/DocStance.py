
import tensorflow as tf
from misc_lib import *
from models.PSF import get_relevant_docs
from nltk import sent_tokenize
from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair
from data_generator.text_encoder import SubwordTextEncoder, SEP_ID, CLS_ID
from data_generator.argmining.ukp import  all_topics
from data_generator.common import get_encoder
from cache import load_from_pickle
from collections import Counter

def get_stance_label(topic):
    return load_from_pickle("stance_{}_rel.pickle".format(topic))

def get_stance_label_ex(topic, blind_topic):
    name = "DocStanc_targ_{}_blind_{}".format(topic, blind_topic)
    return load_from_pickle(name)


class DataLoader:
    def __init__(self, blind_topic, topic_list, max_sequence, input_type):


        self.train_docs = []
        self.dev_docs = []
        split = 10
        for topic in topic_list:
            raw_docs = get_relevant_docs(topic)
            sent_labels = get_stance_label_ex(topic, blind_topic)
            labels = self.pack_sent_labels(sent_labels)
            assert len(raw_docs) == len(labels)


            result = []
            for doc, label in zip(raw_docs, labels):
                e = sent_tokenize(doc), label, topic
                result.append(e)
            self.train_docs.extend(result[split:])
            self.dev_docs.extend(result[:split])



        self.blind_topic = blind_topic
        self.input_type = input_type
        self.max_seq = max_sequence

    @staticmethod
    def pack_sent_labels(sent_labels):
        result = []
        for labels in sent_labels:
            count = Counter(labels)
            total = len(labels)
            probs = list([count[j]/total for j in range(3)])
            result.append(probs)
        return result

    @staticmethod
    def generate_insts(doc, label_list):
        for i, sent in enumerate(doc):
            prev = " ".join(doc[:i])
            if len(prev) < 50:
                continue
            label = label_list[i]
            if label != 0:
                yield prev, label


    def get_train_data(self):
        return self.gen_data(self.train_docs)

    def get_dev_data(self):
        return self.gen_data(self.dev_docs)

    @classmethod
    def gen_data_inner(cls, doc_label_list, max_seq):
        encoder = get_encoder()
        result = []
        for doc, label in doc_label_list:
            pairs = cls.generate_insts(doc, label)
            for sent, label in pairs:
                x1, x2, x3 = cls.encode(sent, encoder, max_seq)
                result.append((x1, x2, x3, label))
        return result

    def gen_data(self, docs):
        result = []
        encoder = get_encoder()
        for i, entry in enumerate(docs):
            doc, label, topic = entry
            if self.input_type == "topic":
                sent = doc[0]
                topic_str = topic + " is good."
                x1, x2, x3 = self.encode_pair(encoder, topic_str, sent)
            else:
                if self.input_type == "first":
                    sent = doc[0]
                elif self.input_type == "last":
                    sent = doc[-1]
                elif self.input_type == "begin":
                    sent = "\n ".join(doc)
                else:
                    assert False
                x1, x2, x3 = self.encode(sent, encoder, self.max_seq)
            result.append((x1, x2, x3, label))

        return result

    @staticmethod
    def encode(text_a, encoder, max_seq):
        tokens_a = encoder.encode(text_a)
        tokens_b = []
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq - 3)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []

        CLS_ID2 = encoder.encode("[CLS]")[0]

        tokens.append(CLS_ID2)
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(SEP_ID)
        segment_ids.append(0)


        input_ids = tokens

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq
        assert len(input_mask) == max_seq
        assert len(segment_ids) == max_seq

        return input_ids, input_mask, segment_ids



    def encode_pair(self, encoder, text_a, text_b):
        tokens_a = encoder.encode(text_a)
        tokens_b = encoder.encode(text_b)

        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, self.max_seq - 3)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append(CLS_ID)
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(SEP_ID)
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append(SEP_ID)
            segment_ids.append(1)

        input_ids = tokens

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq
        assert len(input_mask) == self.max_seq
        assert len(segment_ids) == self.max_seq
        return input_ids, input_mask, segment_ids

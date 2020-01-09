import pickle

from nltk import sent_tokenize

from cpath import data_path
from data_generator.common import get_encoder
from data_generator.text_encoder import SEP_ID
from data_generator.tokenizer_b import _truncate_seq_pair
from misc_lib import *
from models.PSF import get_relevant_docs


def get_pseudo_label_path(topic):
    dir_path = os.path.join(data_path, "arg", "pseudo_label")
    label_path = os.path.join(dir_path, topic + ".pickle")
    return label_path


class DataLoader:
    def __init__(self, target_topic, max_sequence):
        raw_docs = get_relevant_docs(target_topic)

        self.all_docs = lmap(sent_tokenize, raw_docs)
        self.target_topic = target_topic

        split = 4
        self.dev_docs = self.all_docs[:split]
        self.train_docs = self.all_docs[split:]

        label_path = get_pseudo_label_path(target_topic)
        labels =  pickle.load(open(label_path, "rb"))

        self.validate_label(self.all_docs, labels)

        self.max_seq = max_sequence

        self.train_labels = labels[split:]
        self.dev_labels = labels[:split]

    @staticmethod
    def validate_label(docs, labels):
        assert len(docs) == len(labels)
        for doc, label in zip(docs, labels):
            assert len(doc) == len(label)

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
        return self.gen_data(self.train_docs, self.train_labels)

    def get_dev_data(self):
        return self.gen_data(self.dev_docs, self.dev_labels)

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


    def gen_data_parallel(self, docs, labels):
        doc_label_list = list(zip(docs, labels))

        def functor(max_seq):
            def fn(x):
                return self.gen_data_inner(x, max_seq)
            return fn
        num_pool = min(30, int(len(doc_label_list) / 10) )
        return parallel_run(doc_label_list, functor(self.max_seq), num_pool)

    def gen_data(self, docs, labels):
        if len(docs) > 20:
            return self.gen_data_parallel(docs, labels)
        else:
            return self.gen_data_simple(docs, labels)

    def gen_data_simple(self, docs, labels):
        result = []
        encoder = get_encoder()
        for i, doc in enumerate(docs):
            pairs = self.generate_insts(doc, labels[i])
            for sent, label in pairs:
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


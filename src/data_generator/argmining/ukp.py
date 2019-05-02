import os
from path import data_path
from data_generator.data_parser import ukp
from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair
from data_generator.text_encoder import SubwordTextEncoder, CLS_ID, SEP_ID
from models.classic.stopword import load_stopwords
from collections import Counter
from misc_lib import *

all_topics = ["abortion", "cloning", "death_penalty", "gun_control",
                       "marijuana_legalization", "minimum_wage", "nuclear_energy", "school_uniforms"]

class DataLoader:
    def __init__(self, target_topic, is_3way=True):

        self.test_topic = target_topic
        self.train_topics = list(set(all_topics) - {target_topic})
        self.all_data = {topic : ukp.load(topic) for topic in all_topics}
        self.labels = ["NoArgument", "Argument_for", "Argument_against"]
        self.is_3way = is_3way

    def annotation2label(self, annot):
        if self.is_3way:
            return self.labels.index(annot)
        else:
            if annot == "NoArgument":
                return 0
            else:
                return 1

    def get_train_data(self):
        train_data = []
        for topic in self.train_topics:
            for entry in self.all_data[topic]:
                if entry['set'] == "train" :
                    x = entry['sentence']
                    y = self.annotation2label(entry['annotation'])
                    train_data.append((x,y))
        return train_data

    def get_dev_data(self):
        dev_data = []
        for entry in self.all_data[self.test_topic]:
            if entry['set'] == "val":
                x = entry['sentence']
                y = self.annotation2label(entry['annotation'])
                dev_data.append((x,y))
        return dev_data


class BertDataLoader(DataLoader):
    def __init__(self, target_topic, is_3way, max_sequence, vocab_filename):
        DataLoader.__init__(self, target_topic, is_3way)

        self.max_seq = max_sequence
        voca_path = os.path.join(data_path, vocab_filename)

        self.lower_case = True
        self.sep_char = "#"
        self.encoder = FullTokenizerWarpper(voca_path)

    def encode(self, x, y, topic):
        topic_str = topic + " is necessary."
        entry = self.encode_pair(topic_str, x)
        return entry["input_ids"], entry["input_mask"], entry["segment_ids"], y

    def get_train_data(self):
        train_data = []
        for topic in self.train_topics:
            for entry in self.all_data[topic]:
                if entry['set'] == "train" :
                    x = entry['sentence']
                    y = self.annotation2label(entry['annotation'])
                    train_data.append(self.encode(x,y, topic))
        return train_data

    def get_dev_data(self):
        dev_data = []
        for entry in self.all_data[self.test_topic]:
            if entry['set'] == "val":
                x = entry['sentence']
                y = self.annotation2label(entry['annotation'])
                dev_data.append(self.encode(x,y, self.test_topic))
        return dev_data

    def encode_pair(self, text_a, text_b):
        tokens_a = self.encoder.encode(text_a)
        tokens_b = self.encoder.encode(text_b)

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

        return {
            "input_ids": input_ids,
            "input_mask":input_mask,
            "segment_ids": segment_ids
        }


class PairedDataLoader(DataLoader):
    def __init__(self, target_topic, is_3way, max_sequence, vocab_filename):
        DataLoader.__init__(self, target_topic, is_3way)

        self.max_seq = max_sequence
        voca_path = os.path.join(data_path, vocab_filename)
        self.stopwords = load_stopwords()
        self.lower_case = True
        self.sep_char = "#"
        self.encoder = FullTokenizerWarpper(voca_path)

    def encode(self, x1, y1, x2, y2, topic):
        topic_str = topic + " is good."
        entry = self.encode_triple(x1, x2, topic_str)
        # Order is swapped
        return entry["input_ids"], entry["input_mask"], entry["segment_ids"], y1, y2, entry["cls1_idx"], entry["cls2_idx"]

    def get_train_data(self):
        train_data = []
        num_same = 0
        for topic in self.train_topics:
            entries = list([e for e in self.all_data[topic] if e['set'] == "train"])
            for entry in entries:
                x1 = entry['sentence']
                y1 = self.annotation2label(entry['annotation'])
                e2 = self.select_similar(entry, entries)
                x2 = e2['sentence']
                y2 = self.annotation2label(e2['annotation'])
                if y1 == y2:
                    num_same +=1
                train_data.append(self.encode(x1,y1,x2,y2,topic))
        print(" {0:.02f} % are same".format(num_same / len(train_data)))
        return train_data

    def get_repeated_train(self, epoch):
        data = []
        for _ in range(epoch):
            data.extend(self.get_train_data())
        return data

    def get_dev_data(self):
        dev_data = []
        entries = list([e for e in self.all_data[self.test_topic] if e['set'] == "val"])
        for entry in entries:
            x = entry['sentence']
            y = self.annotation2label(entry['annotation'])
            e2 = self.select_similar(entry, entries)
            x2 = e2['sentence']
            y2 = self.annotation2label(e2['annotation'])
            dev_data.append(self.encode(x,y,x2,y2,self.test_topic))
        return dev_data

    def select_similar(self, entry, entries):
        def get_feature(e):
            if "feature" not in e:
                e["feature"] = set(e['sentence'].lower().split()) - self.stopwords
            return e["feature"]

        t1 = get_feature(entry)

        def score(e):
            t2 = get_feature(e)
            return len(t2.intersection(t1))

        entries.sort(key=score, reverse=True)

        e2 = pick1(entries[:50])
        return e2

    def encode_triple(self, text_a, text_b, topic):
        tokens_a = self.encoder.encode(text_a)
        tokens_b = self.encoder.encode(text_b)
        tokens_c = self.encoder.encode(topic)

        def convert(token):
            return self.encoder.encode(token)[0]
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP],[CLS], [SEP] [SEP]with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, self.max_seq - 5 - len(tokens_c))

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
        cls1_idx = len(tokens)
        tokens.append(CLS_ID)
        #tokens.append(convert("[CLS]"))
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(SEP_ID)
        segment_ids.append(0)

        cls2_idx = len(tokens)
        tokens.append(CLS_ID)
        #tokens.append(convert("[CLS]"))
        segment_ids.append(1)
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(SEP_ID)
        segment_ids.append(1)

        for token in tokens_c:
            tokens.append(token)
            segment_ids.append(2)
        tokens.append(SEP_ID)
        segment_ids.append(2)

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

        return {
            "input_ids": input_ids,
            "input_mask":input_mask,
            "segment_ids": segment_ids,
            "cls1_idx": cls1_idx,
            "cls2_idx": cls2_idx,
        }




if __name__ == "__main__":
    d = DataLoader("abortion")
    d.get_train_data()
    d.get_dev_data()

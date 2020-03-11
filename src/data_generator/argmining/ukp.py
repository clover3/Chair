from cache import *
from data_generator.argmining.ukp_header import all_topics
from data_generator.data_parser import ukp
from data_generator.text_encoder import CLS_ID, SEP_ID
from data_generator.tokenizer_wo_tf import FullTokenizerWarpper, _truncate_seq_pair
from misc_lib import *
from models.classic.stopword import load_stopwords
from trainer.tf_module import get_batches_ex


class LazyDict(dict):
    def __init__(self, init_fn):
        super(LazyDict, self).__init__()
        self.init_fn = init_fn
        self.f_init = False

    def keys(self):
        self.do_init()
        return dict.keys(self)

    def __getitem__(self, item):
        self.do_init()
        return dict.__getitem__(self, item)

    def do_init(self):
        if not self.f_init:
            mapf, arg = self.init_fn
            for t in arg:
                self[t] = mapf(t)
            self.f_init = True


class DataLoader:
    def __init__(self, target_topic, is_3way=True):
        self.test_topic = target_topic
        self.train_topics = list(set(all_topics) - {target_topic})
        self.all_data = LazyDict((ukp.load, all_topics))
        self.labels = ["NoArgument", "Argument_for", "Argument_against"]
        self.is_3way = is_3way

    def load_data(self):
        if self.all_data is None:
            self.all_data = {topic: ukp.load(topic) for topic in all_topics}

    def annotation2label(self, annot):
        if self.is_3way:
            return self.labels.index(annot)
        else:
            if annot == "NoArgument":
                return 0
            else:
                return 1

    def get_topic_train_data(self, topic):
        train_data = []
        for entry in self.all_data[topic]:
            if entry['set'] == "train":
                x = entry['sentence']
                y = self.annotation2label(entry['annotation'])
                train_data.append((x, y))
        return train_data

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
    def __init__(self, target_topic, is_3way, max_sequence, vocab_filename, option=""):
        DataLoader.__init__(self, target_topic, is_3way)

        self.max_seq = max_sequence
        voca_path = os.path.join(data_path, vocab_filename)

        self.lower_case = True
        self.sep_char = "#"
        self.encoder = FullTokenizerWarpper(voca_path)
        self.option = option
        self.CLS_ID = CLS_ID
        self.SEP_ID = SEP_ID
        self.weight = 0.0

    @staticmethod
    def expand_topic(t):
        return {"abortion": "abortion, killing",
                "cloning": "cloning , research ",
                "death_penalty": "death penalty , punishment , execution , justice ",
                "gun_control": "gun control , rifles , firearms ",
                "marijuana_legalization": "marijuana legalization , cannabis , drug ",
                "minimum_wage": "minimum wage , labor  , worker ",
                "nuclear_energy": "nuclear energy , power , plant ",
                "school_uniforms": "school uniforms"}[t]

    @staticmethod
    def expand_topic2(t):
        return {"abortion": ["woman", "choice"],
                "cloning": ["research"],
                "death_penalty": ["punishment", "execution", "justice"],
                "gun_control": [ "rifles", "firearms"],
                "marijuana_legalization": ["cannabis", "drug"],
                "minimum_wage": [ "labor", "worker"],
                "nuclear_energy": ["power", "plant"],
                "school_uniforms": []}[t]

    def encode(self, x, y, topic):
        if self.option == "is_good":
            topic_str = topic + " is good."
            entry = self.encode_pair(topic_str, x)
        elif self.option == "only_topic_word":
            topic_str = topic
            #topic_str = expand_topic(topic)
            entry = self.encode_pair(topic_str, x)
        elif self.option == "only_topic_word_reverse":
            topic_str = topic
            #topic_str = expand_topic(topic)
            entry = self.encode_pair(x, topic_str)
        elif self.option == "expand":
            topic_str = self.expand_topic(topic)
            entry = self.encode_pair(topic_str, x)
        elif self.option == "weighted":
            aux_topics = self.expand_topic2(topic)
            entry = self.encode_pair_weighted(topic, aux_topics, x)
        else:
            raise NotImplementedError
        return entry["input_ids"], entry["input_mask"], entry["segment_ids"], y

    def get_train_data(self):
        self.load_data()
        train_data = []
        data_name = "ukp_train_{}_{}_{}".format(self.test_topic, self.option, self.max_seq)
        cached = load_cache(data_name)
        if cached is not None:
            return cached

        for topic in self.train_topics:
            for entry in self.all_data[topic]:
                if entry['set'] == "train" :
                    x = entry['sentence']
                    y = self.annotation2label(entry['annotation'])
                    train_data.append(self.encode(x,y, topic))

        save_to_pickle(train_data, data_name)
        return train_data


    def get_hidden_train_data(self):
        self.load_data()
        h_data = []
        for entry in self.all_data[self.test_topic]:
            if entry['set'] == "train":
                x = entry['sentence']
                y = self.annotation2label(entry['annotation'])
                h_data.append(self.encode(x,y, self.test_topic))
        return h_data



    def get_dev_data(self):
        dev_data = []
        data_name = "ukp_dev_{}_{}_{}".format(self.test_topic, self.option, self.max_seq)
        cached = load_cache(data_name)
        if cached is not None:
            return cached

        for entry in self.all_data[self.test_topic]:
            if entry['set'] == "val":
                x = entry['sentence']
                y = self.annotation2label(entry['annotation'])
                dev_data.append(self.encode(x,y, self.test_topic))
        save_to_pickle(dev_data, data_name)
        return dev_data

    def get_dev_data_expand(self, keyword):

        dev_data = []
        for entry in self.all_data[self.test_topic]:
            if entry['set'] == "val":
                x = entry['sentence']
                y = self.annotation2label(entry['annotation'])
                dev_data.append(self.encode(x,y, keyword))
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
        tokens.append(self.CLS_ID)
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(self.SEP_ID)
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append(self.SEP_ID)
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


    def encode_pair_weighted(self, topic, aux_topics, text_b):
        tokens_topic = self.encoder.encode(topic)
        tokens_aux_topics = list([self.encoder.encode(t) for t in aux_topics])

        tokens_a = tokens_topic + flatten(tokens_aux_topics)

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
        input_mask = []

        tokens.append(CLS_ID)
        input_mask.append(1)
        segment_ids.append(0)

        for idx, token in enumerate(tokens_a):
            tokens.append(token)
            segment_ids.append(0)
            if idx < len(tokens_topic):
                input_mask.append(1)
            else:
                input_mask.append(self.weight)

        tokens.append(SEP_ID)
        segment_ids.append(0)
        input_mask.append(1)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
                input_mask.append(1)
            tokens.append(SEP_ID)
            segment_ids.append(1)
            input_mask.append(1)

        input_ids = tokens

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.

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


class SingleTopicLoader(BertDataLoader):
    def get_train_data(self):
        self.load_data()
        train_data = []
        data_name = "ukp_single_train_{}_{}".format(self.test_topic, self.option)
        cached = load_cache(data_name)
        if cached is not None:
            return cached

        for entry in self.all_data[self.test_topic]:
            if entry['set'] == "train" :
                x = entry['sentence']
                y = self.annotation2label(entry['annotation'])
                train_data.append(self.encode(x,y, self.test_topic))

        save_to_pickle(train_data, data_name)
        return train_data

    def get_dev_data(self):
        dev_data = []
        for entry in self.all_data[self.test_topic]:
            if entry['set'] == "val":
                x = entry['sentence']
                y = self.annotation2label(entry['annotation'])
                dev_data.append(self.encode(x,y, self.test_topic))
        return dev_data


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

class FeedbackData(BertDataLoader):

    def get_train_data(self):
        print("getting feedback data")
        train_data = []
        data_name = "ukp_feedback_{}_{}".format(self.test_topic, self.option)
        cached = load_cache(data_name)
        if cached is not None:
            return cached

        num_classes = 3 if self.is_3way else 2

        raw_data = load_from_pickle("{}_pseudo".format(self.test_topic))

        min_num = min([len(raw_data[key]) for key in raw_data])
        max_num = min_num * 3
        print("Min class: ", min_num)
        count = Counter()
        for label in raw_data:
            for sent in raw_data[label]:
                x = sent
                y = label
                if count[label] < max_num and label < num_classes:
                    train_data.append(self.encode(x,y, self.test_topic))
                    count[label] += 1

        print("Total insts : ", len(train_data))
        save_to_pickle(train_data, data_name)
        return train_data


class NLIAsStance(BertDataLoader):

    def encode(self, x, y, topic):
        statement = {"abortion": "Fetus is a living human.",
         "cloning": "cloning , research ",
         "death_penalty": "death penalty does not prohibit crimes",
         "gun_control": "gun control , rifles , firearms ",
         "marijuana_legalization": "marijuana legalization , cannabis , drug ",
         "minimum_wage": "minimum wage , labor  , worker ",
         "nuclear_energy": "nuclear energy , power , plant ",
         "school_uniforms": "school uniforms"}
        topic_str = statement[topic]
        print(topic_str)
        entry = self.encode_pair(x, topic_str)
        return entry["input_ids"], entry["input_mask"], entry["segment_ids"], y

    def get_dev_data(self):
        dev_data = []
        for entry in self.all_data[self.test_topic]:
            if entry['set'] == "val":
                x = entry['sentence']
                y = self.annotation2label(entry['annotation'])
                dev_data.append(self.encode(x,y, self.test_topic))
        return dev_data


class StreamExplainer(BertDataLoader):
    def __init__(self, target_topic, filepath, is_3way, max_sequence, vocab_filename, option):
        super(StreamExplainer, self).__init__(target_topic, is_3way, max_sequence, vocab_filename, option)
        self.file_path = filepath
        self.line_itr = None
        self.write_idx = 0
        self.read_idx = 0
        self.empty_lines = []
        self.cur_document = []
        self.all_documents = []
        self.lines = open(self.file_path, "r").readlines()
        self.write_last_empty = -1

    def append_write(self, entry):
        assert len(entry) == 2
        while not self.lines[self.write_idx].strip():
            if self.write_last_empty == self.write_idx-1 and self.cur_document:
                self.all_documents.append(self.cur_document)
                self.cur_document = []

            self.write_last_empty = self.write_idx
            self.write_idx += 1

        self.cur_document.append(entry)
        self.write_idx += 1

    def append_dummy(self):
        self.write_idx += 1

    def finish_write(self):
        print("File lines : {} , Written index : {}".format(len(self.lines), self.write_idx))

        self.all_documents.append(self.cur_document)
        print("#Docs : ", len(self.all_documents))

        file_name = self.file_path.split("/")[-1]
        file_name = "b_{}_".format(self.test_topic) + file_name + ".scored"
        save_to_pickle(self.all_documents, file_name)


    def get_data(self):
        y = 0
        line = self.lines[self.read_idx]
        while not line.strip():
            self.empty_lines.append(self.read_idx)
            self.read_idx += 1
            line = self.lines[self.read_idx]
        self.read_idx += 1

        return self.encode(line, y, self.test_topic)

    def get_next_predict_batch(self, batch_size):
        data = []
        try:
            for i in range(batch_size):
                data.append(self.get_data())
        except IndexError as e:
            if not data:
                raise StopIteration()

        batches = get_batches_ex(data, batch_size, 4)
        return batches[0]

    @staticmethod
    def get_second_seg(input_ids):
        for i in range(len(input_ids)):
            if input_ids[i] == SEP_ID:
                idx_sep1 = i
                break
        idx_sep2 = None
        for i in range(idx_sep1 + 1, len(input_ids)):
            if input_ids[i] == SEP_ID:
                idx_sep2 = i
        return idx_sep1+1, idx_sep2




if __name__ == "__main__":
    d = DataLoader("abortion")
    d.get_train_data()
    d.get_dev_data()

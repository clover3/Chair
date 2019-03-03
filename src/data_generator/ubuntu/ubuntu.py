from data_generator.text_encoder import SubwordTextEncoder, CLS_ID, SEP_ID
from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair
import tensorflow as tf
import csv
from path import data_path
from cache import *
from evaluation import *
import unicodedata
from data_generator.NLI.enlidef import *
import copy

num_classes = 2

corpus_dir = os.path.join(data_path, "askubuntu")



class DataLoader:
    def __init__(self, max_sequence, vocab_filename, voca_size, using_alt_tokenizer= False):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.dev_file = os.path.join(corpus_dir, "dev.txt")
        self.test_file = os.path.join(corpus_dir, "test.txt")
        self.max_seq = max_sequence
        self.voca_size = voca_size
        voca_path = os.path.join(data_path, vocab_filename)
        assert os.path.exists(voca_path)
        print(voca_path)

        if not using_alt_tokenizer:
            self.encoder = SubwordTextEncoder(voca_path)
            self.sep_char = "_"
            self.lower_case = False
        else:
            self.lower_case = True
            self.sep_char = "#"
            self.encoder = FullTokenizerWarpper(voca_path)

    def get_train_data(self):
        use_pickle = True
        if use_pickle:
            data = load_from_pickle("ubuntu_train")
        else:
            data = list(self.generate_train_data())
            save_to_pickle(data, "ubuntu_train")
        return data

    def to_triple(self, entry):
        return entry["input_ids"], entry["input_mask"], entry["segment_ids"]

    def generate_train_data(self, interval = None):
        train_label = self.read_train_label()
        text = self.text_reader()

        def get_comb_text(q_id):
            title, body = text[q_id]
            return title + " " + body
        if interval is not None:
            st, ed = interval
            train_label = train_label[st:ed]

        print("train data ", len(train_label))
        timer = TimeEstimator(len(train_label))
        for label_entry in train_label:
            q_id, pos_list, rand_list = label_entry
            q_str = get_comb_text(q_id)
            for pos_id in pos_list:
                pos_text = get_comb_text(pos_id)
                pos_entry = self.encode(q_str, pos_text)
                for neg_id in rand_list:
                    neg_text = get_comb_text(neg_id)
                    neg_entry = self.encode(q_str, neg_text)

                    yield self.to_triple(pos_entry)
                    yield self.to_triple(neg_entry)
            timer.tick()

    def get_dev_data(self):
        if self.dev_data is None:
            self.dev_data = list(self.eval_generator(self.dev_file))
        return self.dev_data

    def flatten_payload(self, data):
        result =[]
        golds = []
        for query_set in data:
            payload, label = query_set

            golds.append(label)
            for p in payload:
                a,b,c = self.to_triple(p)
                result.append((a,b,c))
        return result, golds

    def get_test_data(self):
        if self.test_data is None:
            self.test_data = list(self.eval_generator(self.test_file))
        return self.test_data

    def eval_generator(self, file_path):
        text = self.text_reader()

        def get_comb_text(q_id):
            title, body = text[q_id]
            return title + " " + body

        def parse_list(list_str):
            return list([int(inst) for inst in list_str.split()])

        def get_text(list_str):
            return list([get_comb_text(id) for id in parse_list(list_str)])

        for idx, line in enumerate(tf.gfile.Open(file_path, "rb")):
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            q_id, pos_list, neg_list, bm25_scores = split_line

            q_text = get_comb_text(int(q_id))
            pos_text_list = get_text(pos_list)
            neg_text_list = get_text(neg_list)
            label = [1] * len(pos_text_list) + [0] * len(neg_text_list)
            payload = list([self.encode(q_text, t) for t in pos_text_list + neg_text_list])

            yield payload, label


    def convert_index_out(self, raw_sentence, subtoken_ids, target_idx):
        if self.lower_case:
            raw_sentence = raw_sentence.lower()
        #print("-------")
        #print("raw_sentence", raw_sentence)
        #print("subtoken_ids", subtoken_ids)
        #print("target_idx", target_idx)
        tokens = raw_sentence.split()
        subword_tokens = self.encoder.decode_list(subtoken_ids)
        #print("subword_tokens", subword_tokens)
        #print("target subword", subword_tokens[target_idx])
        if subword_tokens[target_idx].replace("_", "").replace(" ", "") == "":
            target_idx = target_idx - 1
            #print("Replace target_idx to previous", subword_tokens[target_idx])
        prev_text = "".join(subword_tokens[:target_idx])
        text_idx = 0
        #print("prev text", prev_text)
        # now we want to find a token from raw_sentence which appear after prev_text equivalent

        def update_text_idx(target_char, text_idx):
            while prev_text[text_idx] in [self.sep_char, " "]:
                text_idx += 1
            if target_char == prev_text[text_idx]:
                text_idx += 1
            return text_idx

        try:
            for t_idx, token in enumerate(tokens):
                for c in token:
                    # Here, previous char should equal prev_text[text_idx]
                    text_idx = update_text_idx(c, text_idx)
                    # Here, c should equal prev_text[text_idx-1]
                    assert c == prev_text[text_idx-1]

        except IndexError:
            #print("target_token", tokens[t_idx])
            #print("t_idx", t_idx)
            return t_idx
        raise Exception

    def convert_indice_in(self, tokens, input_x, indice, seg_idx):
        sub_tokens = self.split_p_h(input_x[0], input_x)
        subword_tokens = self.encoder.decode_list(sub_tokens[seg_idx])
        start_idx = [1, 1 + len(sub_tokens[0]) + 1][seg_idx]
        in_segment_indice = translate_index(tokens, subword_tokens, indice)
        return list([start_idx + idx for idx in in_segment_indice])

    def class_labels(self):
        return ["similar","not-similar"]

    def text_reader(self):
        filename = "text_tokenized.txt"
        file_path = os.path.join(corpus_dir, filename)
        data = dict()
        for idx, line in enumerate(tf.gfile.Open(file_path, "rb")):
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            if len(split_line) == 3:
                id, title, body = split_line
            else:
                id, title = split_line
                body = ""

            data[int(id)] = (title, body)
        return data

    def read_train_label(self):
        filename = "train_random.txt"
        file_path = os.path.join(corpus_dir, filename)

        def parse_list(list_str):
            return list([int(inst) for inst in list_str.split()])

        data  = []
        for idx, line in enumerate(tf.gfile.Open(file_path, "rb")):
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            q_id, pos_list, rand_list = split_line

            entry = int(q_id), parse_list(pos_list), parse_list(rand_list)
            data.append(entry)
        return data


    # split the np_arr, which is an attribution scores
    @staticmethod
    def split_p_h(np_arr, input_x):
        input_ids, _, seg_idx = input_x
        return DataLoader.split_p_h_with_input_ids(np_arr, input_ids)

    @staticmethod
    def split_p_h_with_input_ids(np_arr, input_ids):

        for i in range(len(input_ids)):
            if input_ids[i] == SEP_ID:
                idx_sep1 = i
                break

        p = np_arr[1:idx_sep1]
        for i in range(idx_sep1 + 1, len(input_ids)):
            if input_ids[i] == SEP_ID:
                idx_sep2 = i
        h = np_arr[idx_sep1 + 1:idx_sep2]
        return p, h

    def encode(self, text_a, text_b):
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


def batch_encode(idx):
    st = idx * 500
    obj = load_from_pickle("ubuntu_train_{}".format(st))
    assert len(obj) % 2 == 0
    data_loader = DataLoader(512, "bert_voca.txt", True)
    batch_size = 32
    from trainer.tf_module import get_batches_ex
    batches = get_batches_ex(obj, batch_size, 3)
    save_to_pickle(batches, "ubuntu_train_batch32_{}".format(idx))



def gen_ubuntu_data_part(interval):
    # dev_data = data_loader.get_dev_data()
    # print("dev num query : ", len(dev_data))
    data_loader = DataLoader(512, "bert_voca.txt", True)

    st, ed = interval
    train_data_piece = list(data_loader.generate_train_data(interval))
    save_to_pickle(train_data_piece, "ubuntu_train_{}".format(st))
    return train_data_piece
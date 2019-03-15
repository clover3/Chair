
from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair, EncoderUnit
from data_generator.text_encoder import SubwordTextEncoder, TokenTextEncoder, CLS_ID, SEP_ID, EOS_ID
from data_generator.NLI import nli
import tensorflow as tf
import csv
from path import data_path
from evaluation import *
num_classes = 3
import random
from data_generator.data_parser.trec import *

scope_dir = os.path.join(data_path, "controversy")
corpus_dir = os.path.join(scope_dir, "title")


def load_annotation():
    path = os.path.join(corpus_dir, "annotate1.csv")
    f = open(path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter=',')


    text_pickle_path = os.path.join(corpus_dir, "dbpedia_controversy_docs.pickle")
    texts_dict = pickle.load(open(text_pickle_path, "rb"))

    def parse_indice(indice):
        return list([int(t) for t in indice.strip().split()])


    for idx, row in enumerate(reader):
        if idx == 0: continue  # skip header
        # Works for both splits even though dev has some extra human labels.
        id, title_indice, desc_indice = row
        title, desc = texts_dict[int(id)]

        yield desc, parse_indice(title_indice), parse_indice(desc_indice)


class DataLoader:
    def __init__(self, max_sequence, vocab_filename, voca_size, is_span):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        voca_path = os.path.join(data_path, vocab_filename)
        self.encoder = SubwordTextEncoder(voca_path)


        self.lower_case = True
        self.sep_char = "#"
        self.encoder = FullTokenizerWarpper(voca_path)
        self.voca_size = voca_size
        self.dev_explain = None
        self.encoder_unit = EncoderUnit(max_sequence, voca_path)
        self.max_seq = max_sequence

        self.question = ["What is title of the controversy?",
                  "What is the controversy about?"]
        if not is_span:
            self.q_id = 0
        else:
            self.q_id = 1
        self.is_span = is_span
        self.text_offset = len(self.encoder.encode(self.question[self.q_id])) + 2

        data = load_annotation()
        self.all_data = self.generate_data(data)
        self.train_data, self.dev_data = self.held_out(self.all_data)


    def get_train_data(self):
        return self.train_data

    def get_dev_data(self):
        return self.dev_data

    def generate_data(self, data):
        result = []
        for entry in data:
            desc, title_indice, desc_indice = entry
            enc_entry = self.encode(desc, self.question[self.q_id])
            indice = [title_indice, desc_indice][self.q_id]

            new_indice = self.translate(desc, indice)
            if self.is_span:
                begin = np.zeros([self.max_seq], dtype=np.int32)
                end = np.zeros([self.max_seq], dtype=np.int32)
                if len(new_indice) > 0:
                    begin_idx = new_indice[0] + self.text_offset
                    end_idx = new_indice[-1] + self.text_offset
                    if begin_idx < self.max_seq:
                        begin[begin_idx] = 1
                        end[min(end_idx,self.max_seq-1)] = 1
                        line = enc_entry['input_ids'], enc_entry['input_mask'], enc_entry['segment_ids'], begin, end
                        result.append(line)
                else:
                    begin[0] = 1
                    end[0] = 1
                    line = enc_entry['input_ids'], enc_entry['input_mask'], enc_entry['segment_ids'], begin, end
                    result.append(line)
            else:
                y = np.zeros([self.max_seq], dtype=np.int32)
                for idx in new_indice:
                    if idx+self.text_offset < self.max_seq:
                        y[idx + self.text_offset] = 1


                line = enc_entry['input_ids'], enc_entry['input_mask'], enc_entry['segment_ids'], y
                if sum(y) > 0:
                    result.append(line)
        return result

    def translate(self, text, indice):
        sw_tokens = self.encoder.decode_list(self.encoder.encode(text))
        parse_tokens = text.split()
        return nli.translate_index(parse_tokens, sw_tokens, indice)

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


    def held_out(self, data):
        heldout_size = int(len(data) * 0.1)
        dev_indice = set(random.sample(range(0, len(data)),  heldout_size))

        train_data = []
        dev_data = []
        for idx, entry in enumerate(data):
            if idx not in dev_indice:
                train_data.append(entry)
            else:
                dev_data.append(entry)

        return train_data, dev_data


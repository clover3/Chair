from data_generator.text_encoder import SubwordTextEncoder, TokenTextEncoder, CLS_ID, SEP_ID, EOS_ID
from data_generator.tokenizer_b import FullTokenizerWarpper
import tensorflow as tf
import six
import os
import csv
from path import data_path
from cache import *
num_classes = 3

corpus_dir = os.path.join(data_path, "nli")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

class DataLoader:
    def __init__(self, max_sequence, vocab_filename, using_alt_tokenizer= False):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.train_file = os.path.join(corpus_dir, "train.tsv")
        self.dev_file = os.path.join(corpus_dir, "dev_matched.tsv")
        self.max_seq = max_sequence
        voca_path = os.path.join(data_path, vocab_filename)
        assert os.path.exists(voca_path)
        print(voca_path)

        if not using_alt_tokenizer:
            self.encoder = SubwordTextEncoder(voca_path)
        else:
            self.encoder = FullTokenizerWarpper(voca_path)


    def get_train_data(self):
        if self.train_data is None:
            self.train_data = list(self.example_generator(self.train_file))
        return self.train_data

    def get_dev_data(self):
        if self.dev_data is None:
            self.dev_data = list(self.example_generator(self.dev_file))
        return self.dev_data

    def get_dev_explain(self):
        explain_data = load_mnli_explain()

        def entry2inst(raw_entry):
            entry = self.encode(raw_entry['p'].lower(), raw_entry['h'].lower())
            return entry["input_ids"], entry["input_mask"], entry["segment_ids"]

        encoded_data = list([entry2inst(entry) for entry in explain_data])
        return encoded_data, explain_data

    def convert_index(self, raw_sentence, subtoken_ids, target_idx):
        #print("raw_sentence", raw_sentence)
        #print("subtoken_ids", subtoken_ids)
        #print("target_idx", target_idx)
        tokens = raw_sentence.split()
        subword_tokens = self.encoder.decode_list(subtoken_ids)
        #print("subword_tokens", subword_tokens)
        #print("target subword", subword_tokens[target_idx])
        prev_text = "".join(subword_tokens[:target_idx])
        text_idx = 0

        try:
            for t_idx, t in enumerate(tokens):
                for c in t:
                    if c == prev_text[text_idx]:
                        text_idx += 1
                    elif prev_text[text_idx] == "_":
                        text_idx += 1
                        if c == prev_text[text_idx]:
                            text_idx += 1
        except IndexError:
            #print("target_token", tokens[t_idx])
            #print("t_idx", t_idx)
            return t_idx
        return t_idx

    def test(self):
        sent = "Nonautomated First-Class and Standard-A mailers cannot ask for their mail to be processed by hand because it costs the postal service more."
        subtoken_ids = self.encoder.encode(sent)
        print(self.encoder.decode_list(subtoken_ids))

    def class_labels(self):
        return ["entailment", "neutral", "contradiction",]

    def example_generator(self, filename):
        label_list = self.class_labels()
        for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Works for both splits even though dev has some extra human labels.
            s1, s2 = split_line[8:10]
            l = label_list.index(split_line[-1])
            entry = self.encode(s1, s2)

            yield entry["input_ids"], entry["input_mask"], entry["segment_ids"], l

    def split_p_h(self, np_arr, input_x):
        input_ids, _, seg_idx = input_x

        for i in range(len(input_ids)):
            if input_ids[i] == SEP_ID:
                idx_sep1 = i
                break

        p = np_arr[1:idx_sep1]
        for i in range(idx_sep1+1, len(input_ids)):
            if input_ids[i] == SEP_ID:
                idx_sep2 = i
        h = np_arr[idx_sep1+1:idx_sep2]
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


def load_nli_explain():
    path = os.path.join(corpus_dir, "nli explain.csv")
    f = open(path, "r")
    reader = csv.reader(f, delimiter=',')

    for idx, row in enumerate(reader):
        if idx ==0 : continue
        premise = row[0]
        hypothesis= row[1]
        tokens_premise = row[2].split()
        tokens_hypothesis= row[3].split()

        for t in tokens_hypothesis:
            if t.lower() not in hypothesis.lower():
                raise Exception(t)
        for t in tokens_premise:
            if t.lower() not in premise.lower():
                print(premise)
                raise Exception(t)

        yield premise, hypothesis, tokens_premise, tokens_hypothesis


def load_nli(path):
    label_list = ["entailment", "neutral", "contradiction", ]

    for idx, line in enumerate(tf.gfile.Open(path, "rb")):
        if idx == 0: continue  # skip header
        line = line.strip().decode("utf-8")
        split_line = line.split("\t")
        s1, s2 = split_line[8:10]
        l = label_list.index(split_line[-1])
        yield s1, s2, l

def load_mnli_explain():
    return load_from_pickle("mnli_explain")
    explation = load_nli_explain()
    dev_file = os.path.join(corpus_dir, "dev_matched.tsv")
    mnli_data = load_nli(dev_file)

    def find(prem, hypothesis):
        for datum in mnli_data:
            s1, s2, l = datum
            if prem == s1.strip() and hypothesis == s2.strip():
                return datum
        print("Not found")
        raise Exception(prem)

    def token_match(tokens1, tokens2):
        gold_indice = []
        for token in tokens1:
            matches = []
            alt_token = [token+".", token+",", token[0].upper() + token[1:]]
            for idx, t in enumerate(tokens2):
                if token == t:
                    matches.append(idx)
                elif t in alt_token:
                    matches.append(idx)

            if len(matches) == 1:
                gold_indice.append(matches[0])
            else:
                for idx, t in enumerate(tokens2):
                    print((idx, t), end =" ")
                print("")
                print(token)
                print(matches)
                print("Select indice: " , end="")
                user_written = input()
                gold_indice += [int(t) for t in user_written.split()]
        return gold_indice

    data = []
    for entry in explation:
        p, h, pe, he = entry

        datum = find(p.strip(),h.strip())
        s1, s2, l = datum

        s1_tokenize = s1.split()
        s2_tokenize = s2.split()

        e_indice_p = token_match(pe, s1_tokenize)
        e_indice_h = token_match(he, s2_tokenize)

        data.append({
            'p': s1,
            'p_tokens': s1_tokenize,
            'h': s2,
            'h_tokens': s2_tokenize,
            'y': l,
            'p_explain':e_indice_p,
            'h_explain':e_indice_h
        })
    save_to_pickle(data, "mnli_explain")
    return data

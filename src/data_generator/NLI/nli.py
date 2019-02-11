from data_generator.text_encoder import SubwordTextEncoder, TokenTextEncoder, CLS_ID, SEP_ID, EOS_ID
from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair
import tensorflow as tf
import six
import os
import csv
from path import data_path
from cache import *
from evaluation import *
import unicodedata

num_classes = 3

corpus_dir = os.path.join(data_path, "nli")




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
            self.sep_char = "_"
            self.lower_case = False
        else:
            self.lower_case = True
            self.sep_char = "#"
            self.encoder = FullTokenizerWarpper(voca_path)

        self.dev_explain_0 = None
        self.dev_explain_1 = None


    def get_train_data(self):
        if self.train_data is None:
            self.train_data = list(self.example_generator(self.train_file))
        return self.train_data

    def get_dev_data(self):
        if self.dev_data is None:
            self.dev_data = list(self.example_generator(self.dev_file))
        return self.dev_data

    def get_train_infos(self):
        infos = list(self.info_generator(self.train_file))
        return infos

    def get_dev_explain_infos(self):
        return list(self.info_generator(self.dev_file))

    def get_dev_explain(self, target):
        if target == 'conflict':
            return self.get_dev_explain_0()
        elif target == 'match':
            return self.get_dev_explain_1(target)
        else:
            assert False

    def get_dev_explain_0(self):
        if self.dev_explain_0 is None:
            explain_data = load_mnli_explain_0()

            def entry2inst(raw_entry):
                entry = self.encode(raw_entry['p'], raw_entry['h'])
                return entry["input_ids"], entry["input_mask"], entry["segment_ids"]

            encoded_data = list([entry2inst(entry) for entry in explain_data])
            self.dev_explain_0 = encoded_data, explain_data

        return self.dev_explain_0

    def get_dev_explain_1(self, tag):
        if self.dev_explain_1 is None:
            explain_data = list(load_nli_explain_1(tag))

            def entry2inst(raw_entry):
                entry = self.encode(raw_entry[0], raw_entry[1])
                return entry["input_ids"], entry["input_mask"], entry["segment_ids"]

            encoded_data = list([entry2inst(entry) for entry in explain_data])
            self.dev_explain_1 = encoded_data, explain_data
        return self.dev_explain_1

    # To compare, concatenate the model's tokens with space removed
    #
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

    def info_generator(self, filename):
        label_list = self.class_labels()
        for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Works for both splits even though dev has some extra human labels.
            s1, s2 = split_line[8:10]
            bp1, bp2 = split_line[4:6] # bp : Binary Parse
            l = label_list.index(split_line[-1])
            yield s1, s2, bp1, bp2

    # split the np_arr, which is an attribution scores
    @staticmethod
    def split_p_h(np_arr, input_x):
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


def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

# Output : Find indice of subword_tokens that covers indice of parse_tokens
# Lowercase must be processed
# indice is for parse_tokens
def translate_index(parse_tokens, subword_tokens, indice):
    print_buf = ""
    result = []
    def dbgprint(s):
        nonlocal print_buf
        print_buf += s + "\n"
        
    try:
        sep_char = "#"
        def normalize_pt(text):
            r = text.replace("``", "\"").replace("''", "\"").replace("`", "'").replace("”", "\"")  \
                .replace("-lrb-", "(").replace("-rrb-", ")")\
                .replace("-lsb-", "[").replace("-rsb-", "]")\
                .replace("…", "...")\
                .replace("«", "\"")
            #    .replace("&", "&amp;")
            return _run_strip_accents(r)

        def normalize_pt_str(s):
            #if "&amp;" not in s and "& amp" not in s:
            #    s = s.replace("&", "&amp;")
            return s

        def normalize_sw(text):
            s = text.replace("”", "\"").replace("“", "\"").replace("…", "...").replace("«", "\"")
            return s


        parse_tokens = list([normalize_pt(t.lower()) for t in parse_tokens])
        subword_tokens = list([normalize_sw(t) for t in subword_tokens])
        dbgprint("----")
        dbgprint("parse_tokens : " + " ".join(parse_tokens))
        dbgprint("subword_tokens : " + " ".join(subword_tokens))
        for target_index in indice:
            if target_index > len(subword_tokens):
                break
            pt_begin = normalize_pt_str("".join(parse_tokens[:target_index]))

            pt_idx = 0
            dbgprint("Target_index {}".format(target_index))
            dbgprint("prev_text : " + pt_begin)

            # find the index in subword_tokens that covers

            # Step 1 : Find begin of parse_tokens[target_idx]
            swt_idx = 0
            sw_idx = 0
            while pt_idx < len(pt_begin):
                token = subword_tokens[swt_idx]
                if sw_idx == 0:
                    dbgprint(token)

                c = token[sw_idx]
                if c in [sep_char, " "]:
                    sw_idx += 1
                elif c == pt_begin[pt_idx]:
                    sw_idx += 1
                    pt_idx += 1
                    assert c == pt_begin[pt_idx - 1]
                else:
                    raise Exception("Non matching 1 : {} not equal {}".format(c, pt_begin[pt_idx]) )
                    assert False

                if sw_idx == len(token): # Next token
                    swt_idx += 1
                    sw_idx = 0

            while subword_tokens[swt_idx][sw_idx] in [sep_char, " "]:
                sw_idx += 1

            dbgprint("")
            if subword_tokens[swt_idx] == "[UNK]":
                continue
            assert pt_idx == len(pt_begin)
            if not parse_tokens[target_index][0] == subword_tokens[swt_idx][sw_idx]:
                print("swt_idx = {} sw_idx= {}".format(swt_idx, sw_idx))
                raise Exception("Non matching 2 : {} not equal {}".format(parse_tokens[target_index][0], subword_tokens[swt_idx][sw_idx]) )
            # Step 2 : Add till parse_tokens[target_idx] ends
            pt_end = normalize_pt_str("".join(parse_tokens[:target_index + 1]))
            dbgprint("pt_end : " + pt_end)
            while pt_idx < len(pt_end):
                token = subword_tokens[swt_idx]
                if len(result) == 0 or result[-1] != swt_idx:
                    dbgprint("Append {} ({})".format(swt_idx, token))
                    result.append(swt_idx)

                c = token[sw_idx]
                if c in [sep_char, " "]:
                    sw_idx += 1
                elif c == pt_end[pt_idx]:
                    sw_idx += 1
                    pt_idx += 1
                    assert c == pt_end[pt_idx - 1]
                else:
                    raise Exception("Non matching 3 : {} not equal {}".format(c, pt_begin[pt_idx]))
                    assert False

                if sw_idx == len(token): # Next token
                    swt_idx += 1
                    sw_idx = 0
    except Exception as e:
        print(e)
        print(print_buf)
    return result


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


def load_nli_explain_1(name):

    path_idx = os.path.join(corpus_dir, "{}.txt".format(name))
    path_text = os.path.join(corpus_dir, "{}.csv".format(name))

    reader = csv.reader(open(path_text, "r"), delimiter=',')

    texts_list = []
    for row in reader:
        premise = row[0]
        hypothesis = row[1]
        texts_list.append((premise, hypothesis))

    f = open(path_idx, "r")
    indice_list = []
    for line in f:
        p_indice, h_indice = line.split(",")
        p_indice = list([int(t) for t in p_indice.split(" ")])
        h_indice = list([int(t) for t in h_indice.split(" ")])
        indice_list.append((p_indice, h_indice))

    def complement(source, whole):
        return list(set(whole) - set(source))

    texts_list = texts_list[:len(indice_list)]
    debug = False
    for (prem, hypo), (p_indice, h_indice) in zip(texts_list, indice_list):
        p_tokens = prem.split()
        h_tokens = hypo.split()
        if name == "align":
            p_indice = complement(p_indice, range(len(p_tokens)))
            h_indice = complement(h_indice, range(len(h_tokens)))
        if debug:
            print(len(p_tokens), len(p_indice),len(h_tokens), len(h_indice))
            for idx in p_indice:
                print(p_tokens[idx], end=" ")
            print(" | ", end="")
            for idx in range(len(p_tokens)):
                if idx not in p_indice:
                    print(p_tokens[idx], end=" ")
            print("")
            for idx in h_indice:
                print(h_tokens[idx], end=" ")
            print(" | ", end="")
            for idx in range(len(h_tokens)):
                if idx not in h_indice:
                    print(h_tokens[idx], end=" ")
            print("")
        yield prem, hypo, p_indice, h_indice

def load_nli(path):
    label_list = ["entailment", "neutral", "contradiction", ]

    for idx, line in enumerate(tf.gfile.Open(path, "rb")):
        if idx == 0: continue  # skip header
        line = line.strip().decode("utf-8")
        split_line = line.split("\t")
        s1, s2 = split_line[8:10]
        l = label_list.index(split_line[-1])
        yield s1, s2, l

def load_mnli_explain_0():
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


def eval_explain(conf_score, data_loader, tag):
    if tag == 'conflict':
        return eval_explain_0(conf_score, data_loader)
    else:
        return eval_explain_1(conf_score, data_loader, tag)


def eval_explain_1(conf_score, data_loader, tag):
    enc_explain_dev, explain_dev = data_loader.get_dev_explain_1(tag)

    pred_list = []
    gold_list = []
    for idx, entry in enumerate(explain_dev):
        conf_p, conf_h = data_loader.split_p_h(conf_score[idx], enc_explain_dev[idx])
        prem, hypo, p_indice, h_indice = entry
        input_ids = enc_explain_dev[idx][0]
        p_enc, h_enc = data_loader.split_p_h(input_ids, enc_explain_dev[idx])

        p_explain = []
        h_explain = []

        for i in top_k_idx(conf_p, 10):
            # Convert the index of model's tokenization into space tokenized index
            v_i = data_loader.convert_index_out(prem, p_enc, i)
            score = conf_p[i]
            p_explain.append((score, v_i))

        for i in top_k_idx(conf_h, 10):
            v_i = data_loader.convert_index_out(hypo, h_enc, i)
            score = conf_h[i]
            h_explain.append((score, v_i))

        pred_list.append((p_explain, h_explain))
        gold_list.append((p_indice, h_indice))

    p_at_1 = p_at_k_list(pred_list, gold_list, 1)
    MAP_score = MAP(pred_list, gold_list)
    return p_at_1, MAP_score


def eval_explain_0(conf_score, data_loader):
    enc_explain_dev, explain_dev = data_loader.get_dev_explain_0()

    pred_list = []
    gold_list = []
    for idx, entry in enumerate(explain_dev):
        attr_p, attr_h = data_loader.split_p_h(conf_score[idx], enc_explain_dev[idx])

        input_ids = enc_explain_dev[idx][0]
        p, h = data_loader.split_p_h(input_ids, enc_explain_dev[idx])

        p_explain = []
        for i in top_k_idx(attr_p, 10):
            v_i = data_loader.convert_index_out(entry['p'], p, i)
            score = attr_p[i]
            p_explain.append((score, v_i))
        h_explain = []
        for i in top_k_idx(attr_h, 10):
            v_i = data_loader.convert_index_out(entry['h'], h, i)
            score = attr_h[i]
            h_explain.append((score, v_i))

        pred_list.append((p_explain, h_explain))
        gold_list.append((entry['p_explain'], entry['h_explain']))

    p_at_1 = p_at_k_list(pred_list, gold_list, 1)
    MAP_score = MAP(pred_list, gold_list)
    return p_at_1, MAP_score



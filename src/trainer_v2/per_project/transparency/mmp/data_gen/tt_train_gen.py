import math
from collections import Counter, OrderedDict
from typing import Dict

from transformers import AutoTokenizer
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from tlm.data_gen.bert_data_gen import create_int_feature, create_float_feature
from trainer_v2.per_project.transparency.transformers_utils import pad_truncate
from typing import List, Iterable, Callable, Dict, Tuple, Set


def get_convert_to_bow():
    word_tokenizer = KrovetzNLTKTokenizer()
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_subword_per_word = 4
    max_terms = 100
    max_total = max_subword_per_word * max_terms
    def convert_to_bow(text):
        word_tokens = word_tokenizer.tokenize_stem(text)
        tf_d = Counter(word_tokens)
        terms, cnt_s = zip(*tf_d.items())
        terms = list(terms)[:max_terms]
        cnt_s = pad_truncate(list(cnt_s), max_terms)
        input_ids_all = []
        for term in terms:
            input_ids = bert_tokenizer(term)["input_ids"]
            input_ids = pad_truncate(input_ids, max_subword_per_word)
            input_ids_all.extend(input_ids)
        input_ids_all = pad_truncate(input_ids_all, max_total)
        return cnt_s, input_ids_all
    return convert_to_bow


def get_convert_to_bow_qtw():
    word_tokenizer = KrovetzNLTKTokenizer()
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    cdf, df_ = load_msmarco_passage_term_stat()
    def term_idf_factor(term):
        N = cdf
        df = df_[term]
        return math.log((N - df + 0.5) / (df + 0.5))

    max_subword_per_word = 4
    max_terms = 100
    max_total = max_subword_per_word * max_terms
    def convert_to_bow(text):
        word_tokens = word_tokenizer.tokenize_stem(text)
        tf_d = Counter(word_tokens)
        terms, cnt_s = zip(*tf_d.items())
        terms = list(terms)
        terms = terms[:max_terms]
        cnt_s = pad_truncate(list(cnt_s), max_terms)
        input_ids_all = []
        for term in terms:
            input_ids = bert_tokenizer(term)["input_ids"]
            input_ids = pad_truncate(input_ids, max_subword_per_word)
            input_ids_all.extend(input_ids)
        input_ids_all = pad_truncate(input_ids_all, max_total)

        qtw = pad_truncate(list(map(term_idf_factor, terms)), max_terms)
        return cnt_s, input_ids_all, qtw
    return convert_to_bow


# TT = Translation Table
# TT-BM25
#    The query term's extended document frequency is computed by NN.
#         For each of (
#

def get_encode_fn_for_word_encoder():
    convert_to_bow = get_convert_to_bow()
    def encode_fn(q_pos_neg: Tuple[str]):
        item = {
            "q": q_pos_neg[0],
            "d1": q_pos_neg[1],
            "d2": q_pos_neg[2],
        }
        feature: OrderedDict = OrderedDict()
        for text_role in ["q", "d1", "d2"]:
            text = item[text_role]
            cnt_s, input_ids_all = convert_to_bow(text)
            feature[f"{text_role}_input_ids"] = create_int_feature(input_ids_all)
            feature[f"{text_role}_tfs"] = create_int_feature(cnt_s)
        return feature

    return encode_fn



def get_encode_fn_for_word_encoder_qtw():
    convert_to_bow = get_convert_to_bow_qtw()
    def encode_fn(q_pos_neg: Tuple[str]):
        item = {
            "q": q_pos_neg[0],
            "d1": q_pos_neg[1],
            "d2": q_pos_neg[2],
        }
        feature: OrderedDict = OrderedDict()
        for text_role in ["q", "d1", "d2"]:
            text = item[text_role]
            cnt_s, input_ids_all, qtw = convert_to_bow(text)
            feature[f"{text_role}_input_ids"] = create_int_feature(input_ids_all)
            feature[f"{text_role}_tfs"] = create_int_feature(cnt_s)
            feature[f"{text_role}_qtw"] = create_float_feature(qtw)
        return feature

    return encode_fn


import random
from collections import OrderedDict
from typing import List, Iterator, NamedTuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from datastore.interface import load_multiple, load_multiple_divided
from datastore.table_names import BertTokenizedCluewebDoc
from exec_lib import run_func_with_config
from galagos.parse import get_doc_ids_from_ranked_list_path
from list_lib import lmap, flatten
# datasize
# 2095 lines
# 1237 docs
# 252 claims
# 470 perspective sentences
from misc_lib import get_duplicate_list, DataIDManager
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature


def doc_hash(doc: List[List[str]]):
    return " ".join(flatten(doc))


def enum_passages(doc: List[List[str]], max_seq_length) -> Iterator[List[str]]:
    cur_passage = []
    for sent in doc:
        if len(cur_passage) + len(sent) <= max_seq_length:
            cur_passage.extend(sent)
        else:
            yield cur_passage
            cur_passage = []

    if cur_passage:
        yield cur_passage


class Instance(NamedTuple):
    tokens: List[str]
    data_id : int
    label: int


def generate(pos_doc_ids, all_doc_list, max_seq_length) -> List[Instance]:
    # load list of documents
    # make list of negative documents.
    # remove duplicates.
    seq_length = max_seq_length - 2
    neg_docs_ids = list([d for d in all_doc_list if d not in pos_doc_ids])
    pos_docs: List[List[List[str]]] = load_multiple(BertTokenizedCluewebDoc, pos_doc_ids, True)
    hashes = lmap(doc_hash, pos_docs)
    duplicate_indice = get_duplicate_list(hashes)
    pos_docs: List[List[List[str]]] = list([doc for i, doc in enumerate(pos_docs) if i not in duplicate_indice])
    neg_docs: List[List[List[str]]] = load_multiple_divided(BertTokenizedCluewebDoc, neg_docs_ids, True)

    data_id_man = DataIDManager()

    def enum_instances(doc_list: List[List[List[str]]], label: int) -> Iterator[Instance]:
        for d in doc_list:
            for passage in enum_passages(d, seq_length):
                yield Instance(passage, data_id_man.assign([]), label)
    pos_insts = list(enum_instances(pos_docs, 1))
    neg_insts = list(enum_instances(neg_docs, 0))
    all_insts = pos_insts + neg_insts
    print("{} instances".format(len(all_insts)))
    random.shuffle(all_insts)
    return all_insts


def encode_w_data_id(tokenizer, max_seq_length, t: Instance):
    tokens = ["[CLS]"] + t.tokens + ["[SEP]"]
    segment_ids = [0] * (len(t.tokens) + 2)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    features = get_basic_input_feature(tokenizer, max_seq_length, tokens, segment_ids)
    features['label_ids'] = create_int_feature([int(t.label)])
    features['data_id'] = create_int_feature([int(t.data_id)])
    return features


def make_training_data(config):
    pos_doc_list_path = config['doc_list_path']
    q_res_path = config['q_res_path']
    save_path = config['save_path']
    balance_test = config['balance_test']

    max_seq_length = 512

    pos_doc_ids = set([l.strip() for l in open(pos_doc_list_path, "r").readlines()])
    doc_ids_unique = get_doc_ids_from_ranked_list_path(q_res_path)

    insts = generate(list(pos_doc_ids), list(doc_ids_unique), max_seq_length)

    train_size = int(0.9 * len(insts))
    train_insts = insts[:train_size]
    val_insts = insts[train_size:]

    val_pos_insts = list([i for i in val_insts if i.label == 1])
    val_neg_insts = list([i for i in val_insts if not i.label])
    print("num pos inst in val", len(val_pos_insts))
    if balance_test:
        val_neg_insts = val_neg_insts[:len(val_pos_insts)]
    val_insts = val_pos_insts + val_neg_insts

    tokenizer = get_tokenizer()

    def encode_fn(inst: Instance) -> OrderedDict:
        return encode_w_data_id(tokenizer, max_seq_length, inst)

    write_records_w_encode_fn(save_path + "train", encode_fn, train_insts)
    write_records_w_encode_fn(save_path + "val", encode_fn, val_insts)


if __name__ == "__main__":
    run_func_with_config(make_training_data)

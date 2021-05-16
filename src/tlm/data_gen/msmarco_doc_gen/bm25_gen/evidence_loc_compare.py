from typing import List, Iterable, Callable, Dict, Tuple, Set

from arg.perspectives.pc_tokenizer import PCTokenizer
from dataset_specific.msmarco.analyze_code.doc_passage_join import get_passage_dict
from dataset_specific.msmarco.common import load_query_group, QueryID
import csv
from collections import defaultdict

from cache import save_to_pickle, load_from_pickle
from cpath import at_output_dir, at_data_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.analyze_code.lcs_imp import split_indexed, lcs
from dataset_specific.msmarco.common import load_train_queries, load_per_query_docs, load_msmarco_simple_qrels, \
    SimpleQrel, load_msmarco_raw_qrels, MSMarcoDoc
from typing import List, Iterable, Callable, Dict, Tuple, Set, NamedTuple

from galagos.types import QueryID
from list_lib import left, get_max_idx, lflatten
from misc_lib import BinHistogram, TimeEstimator
from tab_print import print_table
from tlm.data_gen.msmarco_doc_gen.max_sent_encode import regroup_sent_list
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10docMulti
from trec.qrel_parse import load_qrels_structured
from trec.types import QRelsDict


def get_passages(qids, passage_qrels):
    passage_ids_to_find = []
    for qid in qids:
        for passage_id, score in passage_qrels[qid].items():
            if score:
                passage_ids_to_find.append(passage_id)
    passage_dict = get_passage_dict(passage_ids_to_find)
    return passage_dict


def main():
    split = "train"
    resource = ProcessedResource10docMulti(split)

    query_group: List[List[QueryID]] = load_query_group(split)
    msmarco_passage_qrel_path = at_data_dir("msmarco", "qrels.train.tsv")
    passage_qrels: QRelsDict = load_qrels_structured(msmarco_passage_qrel_path)

    qids = query_group[0]
    qids = qids[:100]
    pickle_name = "msmarco_passage_doc_analyze_passage_dict_evidence_loc"
    try:
        passage_dict = load_from_pickle(pickle_name)
    except FileNotFoundError:
        print("Reading passages...")
        passage_dict = get_passages(qids, passage_qrels)
        save_to_pickle(passage_dict, pickle_name)
    def get_rel_doc_id(qid):
        if qid not in resource.get_doc_for_query_d():
            raise KeyError
        for doc_id in resource.get_doc_for_query_d()[qid]:
            label = resource.get_label(qid, doc_id)
            if label:
                return doc_id
        raise KeyError

    def translate_token_idx_to_sent_idx(stemmed_body_tokens_list, loc_in_body):
        acc = 0
        for idx, tokens in enumerate(stemmed_body_tokens_list):
            acc += len(tokens)
            if loc_in_body < acc:
                return idx
        return -1

    pc_tokenize = PCTokenizer()
    bert_tokenizer = get_tokenizer()

    for qid in qids:
        try:
            doc_id = get_rel_doc_id(qid)
            stemmed_tokens_d = resource.get_stemmed_tokens_d(qid)
            stemmed_title_tokens, stemmed_body_tokens_list = stemmed_tokens_d[doc_id]
            rel_passages = list([passage_id for passage_id, score in passage_qrels[qid].items() if score])
            success = False
            found_idx = -1
            for rel_passage_id in rel_passages:
                passage_text = passage_dict[rel_passage_id].strip()
                passage_tokens = pc_tokenize.tokenize_stem(passage_text)
                stemmed_body_tokens_flat = lflatten(stemmed_body_tokens_list)
                n, log = lcs(passage_tokens, stemmed_body_tokens_flat, True)
                if len(passage_tokens) > 4 and n > len(passage_tokens) * 0.7 and n > 0:
                    success = True
                    _, loc_in_body = log[0]

                    sent_idx = translate_token_idx_to_sent_idx(stemmed_body_tokens_list, loc_in_body)
                    prev = stemmed_body_tokens_flat[:loc_in_body]

                    loc_by_bert_tokenize = len(bert_tokenizer.tokenize(" ".join(prev)))
                    print(sent_idx, loc_in_body, loc_by_bert_tokenize, len(stemmed_body_tokens_list))
                    found_idx = sent_idx
            if not success:
                print("Not found. doc_lines={} passage_len={}".format(len(stemmed_body_tokens_list), len(passage_tokens)))

        except KeyError:
            pass


if __name__ == "__main__":
    main()

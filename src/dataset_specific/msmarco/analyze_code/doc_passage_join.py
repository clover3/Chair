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
from list_lib import left, get_max_idx
from misc_lib import BinHistogram, TimeEstimator
from tab_print import print_table
from trec.qrel_parse import load_qrels_structured
from trec.types import QRelsDict



def heuristic_find(text_pattern, long_text, debug=False):
    tokens = text_pattern.split()
    st_cursor = 0
    mid_cursor = 0
    token_cursor = 0
    grace_max = 2
    tolerance = 0
    found_idx = -1
    while st_cursor < len(long_text):
        if token_cursor > 0:
            token = tokens[token_cursor]
            ed = mid_cursor + len(token) + 30
            idx = long_text.find(token, mid_cursor, ed)
            if debug:
                print(token, idx)
            if idx >= 0:
                # found token
                mid_cursor = idx + len(token)
                token_cursor += 1
                tolerance += 0.3
            elif tolerance < grace_max:
                # not found token
                tolerance -= 1
                token_cursor += 1

            if tolerance <= 0:
                # reset search
                tolerance = 0
                token_cursor = 0
                st_cursor += len(tokens[0])
        else:
            token = tokens[token_cursor]
            idx = long_text.find(token, st_cursor)
            if debug:
                print(token, idx)
            if idx >= 0:
                st_cursor = idx
                mid_cursor = idx + len(token)
                token_cursor += 1
            else:
                break

        if token_cursor == len(tokens) or token_cursor > 20:
            found_idx = st_cursor
            break

    return found_idx


def lcs_based_join_ex(text_pattern, long_text, debug=False):
    idx = long_text.find(text_pattern)
    if idx >= 0:
        return idx 
    else:
        return lcs_based_join(text_pattern, long_text, debug)


def lcs_based_join(text_pattern, long_text, debug=False):
    tokens1, indices1 = split_indexed(text_pattern)
    tokens2, indices2 = split_indexed(long_text)
    n, log = lcs(tokens1, tokens2, True)
    if n > len(tokens1) * 0.6 and n > 0:
        _, token2_start = log[0]
        st, ed = indices2[token2_start]
        return st
    elif n > 4:
        if debug:
            print(n, len(tokens1))
            print(log)
            print(list([tokens1[idx] for idx in left(log)]))
        return -1
    else:
        return -1


class JoinedPassage(NamedTuple):
    id: str
    loc: int
    text: str


def join_doc_passage(todo: List[Tuple[str, MSMarcoDoc]],
                     passage_qrels,
                     passage_dict) -> Iterable[Tuple[str, MSMarcoDoc, JoinedPassage]]:
    print("join_doc_passage")
    fail_cnt = 0
    suc_cnt = 0
    ticker = TimeEstimator(1000)
    for qid, doc in todo:
        rel_passages = list([passage_id for passage_id, score in passage_qrels[qid].items() if score])
        if len(rel_passages) > 1:
            print("{} rel passages found".format(len(rel_passages)))

        any_success = False
        for rel_passage_id in rel_passages:
            passage_text = passage_dict[rel_passage_id].strip()
            passage_loc = lcs_based_join(passage_text, doc.body)
            if passage_loc < 0:
                pass
            #     print("-----------------------")
            #     print("query: {}".format(qid))
            #     print("passage_id:", rel_passage_id)
            #     print("passage:", passage_text)
            #
            #     print("doc_id", doc.doc_id)
            #     print('doc:')
            #     print("doc_len", len(doc.body))
            #     print("passage not found in doc")
            #     print(doc.body)
            # # print(doc.title)
            #     lcs_based_join(passage_text, doc.body, True)
            else:
                any_success = True
                yield qid, doc, JoinedPassage(rel_passage_id, passage_loc, passage_text)
        if not any_success:
            fail_cnt += 1
            print(suc_cnt, fail_cnt)
        else:
            suc_cnt += 1
        ticker.tick()




def get_todo() -> List[Tuple[QueryID, MSMarcoDoc]]:
    print("get_todo()")
    doc_queries = load_train_queries()
    doc_qrels: Dict[QueryID, List[str]] = load_msmarco_raw_qrels("train")

    todo: List[Tuple[QueryID, MSMarcoDoc]] = []
    doc_id_to_find = []
    n_item = 1000
    for qid, q_text in doc_queries[:n_item]:
        docs = load_per_query_docs(qid, None)
        for doc in docs:
            if doc.doc_id in doc_qrels[qid]:
                todo.append((qid, doc))
                doc_id_to_find.append(doc.doc_id)
    return todo


def load_passage_dict(todo, passage_qrels):
    passage_ids_to_find = []
    qids = left(todo)
    for qid in qids:
        for passage_id, score in passage_qrels[qid].items():
            if score:
                passage_ids_to_find.append(passage_id)
    passage_dict = get_passage_dict(passage_ids_to_find)
    save_to_pickle(passage_dict, "msmarco_passage_doc_analyze_passage_dict")
    return passage_dict


def get_statistic_for_join(join_result: Iterable[Tuple[str, MSMarcoDoc, JoinedPassage]]):
    print("get_statistic_for_join()")
    tokenizer = get_tokenizer()

    def size_in_tokens(text):
        return len(tokenizer.tokenize(text))

    intervals = list(range(0, 500, 50)) + list(range(500, 5000, 500))
    last = "5000 <"
    keys = intervals + [last]
    def bin_fn(n):
        for ceil in intervals:
            if n < ceil:
                return ceil

        return "5000 <"

    bin_doc = BinHistogram(bin_fn)
    bin_loc = BinHistogram(bin_fn)
    bin_passage = BinHistogram(bin_fn)

    match_fail = 0
    for doc, passage in join_result:
        if passage.loc >= 0:
            prev = doc.body[:passage.loc]
            n_tokens_before = len(tokenizer.tokenize(prev))
            passage_text = passage.text
            passage_len = len(passage_text)
            # print("passage loc", passage_loc)
            # print(n_tokens_before)
            bin_doc.add(size_in_tokens(doc.body))
            bin_loc.add(size_in_tokens(prev))
            bin_passage.add(size_in_tokens(passage_text))

            # print(prev)
            # print("  >>>>>  ")
            # print(passage_maybe)
            # print("   <<<< ")
            # print(next)
            pass
        else:
            match_fail += 1
            # print("passage not found in doc")
            # print(doc.body)

    print('match fail', match_fail)
    print("doc length")
    bins = [bin_doc, bin_passage, bin_loc]
    head = ['', 'bin_doc', 'bin_passage', 'bin_loc']
    rows = [head]
    for key in keys:
        row = [key]
        for bin in bins:
            row.append(bin.counter[key])
        rows.append(row)

    print_table(rows)


def get_passage_dict(passage_ids_to_find):
    msmarco_passage_corpus_path = at_data_dir("msmarco", "collection.tsv")
    passage_dict = {}
    with open(msmarco_passage_corpus_path, 'r', encoding='utf8') as f:
        for line in f:
            passage_id, text = line.split("\t")
            if passage_id in passage_ids_to_find:
                passage_dict[passage_id] = text
    return passage_dict




def main():
    todo: List[Tuple[QueryID, MSMarcoDoc]] = get_todo()
    msmarco_passage_qrel_path = at_data_dir("msmarco", "qrels.train.tsv")
    passage_qrels: QRelsDict = load_qrels_structured(msmarco_passage_qrel_path)

    try:
        passage_dict = load_from_pickle("msmarco_passage_doc_analyze_passage_dict")
    except FileNotFoundError:
        passage_dict = load_passage_dict(todo, passage_qrels)
    doc_queries = dict(load_train_queries())

    itr: Iterable[Tuple[str, MSMarcoDoc, JoinedPassage]] = join_doc_passage(todo, passage_qrels, passage_dict)
    ##
    for qid, doc, passage in itr:
        query_text = doc_queries[QueryID(qid)]
        print('query', qid, query_text)
        prev = doc.body[:passage.loc]
        passage_text = passage.text
        tail = doc.body[passage.loc + len(passage_text):]
        print("-----")
        print(prev)
        print(">>>")
        print(passage_text)
        print("<<<")
        print(tail)
        print("-----")

    # get_statistic_for_join(itr)


if __name__ == "__main__":
    main()

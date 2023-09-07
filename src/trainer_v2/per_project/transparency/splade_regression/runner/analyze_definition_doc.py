from collections import defaultdict, Counter
from cpath import output_path
from table_lib import tsv_iter
from list_lib import left
from misc_lib import path_join, get_second
from typing import List, Iterable, Callable, Dict, Tuple, Set

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.splade_regression.modeling.splade_predictor import get_splade
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def query_filter(query):
    query_lower = query.lower()

    target = ["what is", "define", "definition"]
    for t in target:
        if t in query_lower:
            return True
    return False


def main():
    c_log.info("loading the splade model")
    strategy = get_strategy()
    with strategy.scope():
        splade = get_splade()
    quad_tsv_path = path_join("data", "msmarco", "sample_dev1000", "corpus.tsv")
    top1000_iter: Iterable[Tuple[str, str]] = tsv_iter(quad_tsv_path)
    c_log.info("iterating data")
    qid_clustered = defaultdict(list)
    for qid, pid, query, text in top1000_iter:
        e = qid, pid, query, text
        if query_filter(query):
            qid_clustered[qid].append(e)

    def get_text_as_tokens(text):
        encoded_input = splade.tokenizer(text)
        return [splade.reverse_voc[t] for t in encoded_input["input_ids"]]
    ##
    c_log.info("Now encoding")
    # score all pairs
    #   P( Term in Emb | Term in D)
    for qid, items in qid_clustered.items():
        query = items[0][2]
        print("Query: {}".format(query))

        text_list = [text for qid, pid, query, text in items]
        q_enc = splade.encode_batch_simple_out([query])[0]
        text_batch = text_list[:100]
        text_enc_list = splade.encode_batch_simple_out(text_batch)
        q_enc.sort(key=get_second, reverse=True)
        for t, score in q_enc:
            counter = Counter()
            print("Query term {0} {1:.1f}".format(t, score))
            for d_enc, d_text in zip(text_enc_list, text_batch):
                d_tokens = get_text_as_tokens(d_text)
                d_enc_keys = left(d_enc)
                # t_surface = t[2:] if t[:2] == "##" else t
                t_in_d_raw = t in d_tokens
                t_in_d_enc = t in d_enc_keys
                if t_in_d_raw and not t_in_d_enc:
                    print(f"{t} is in raw passage but not in enc: {d_text}")
                # if not t_in_d_raw and t_in_d_enc:
                #     print(f"{t} is not in passage but in enc: {d_text}")
                counter[(t_in_d_raw, t_in_d_enc)] += 1
            print(counter)


if __name__ == "__main__":
    main()
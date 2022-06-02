import json
from typing import List, Dict, Tuple, NamedTuple

from nltk import sent_tokenize

from arg.counter_arg_retrieval.f5.load_f5_clue_docs import load_all_docs_cleaned
from cache import load_from_pickle
from cpath import at_output_dir
from list_lib import lmap, get_max_idx
from scipy_aux import logit_to_score_softmax
from tab_print import print_table
from tlm.estimator_output_reader import join_prediction_with_info


def get_f5_tids_score_d_from_bert():
    info_save_path = at_output_dir("clue_counter_arg", "clue_f5.tfrecord.info")
    info = json.load(open(info_save_path, "r"))

    prediction_file = at_output_dir("clue_counter_arg", "ada_aawd5_clue.4000.score")
    pred_data = join_prediction_with_info(prediction_file, info)
    score_d = {}

    for idx, e in enumerate(pred_data):
        score = logit_to_score_softmax(e['logits'])
        text = e['text']
        score_d[text] = score
    return score_d


def get_f5_tids_score_d_from_svm():
    output: List[Tuple[str, float]] = load_from_pickle("f5_svm_aawd_prediction")
    return dict(output)


def main():
    rlp = "C:\\work\\Code\\Chair\\output\\clue_counter_arg\\ranked_list.txt"
    html_dir = "C:\\work\\Code\\Chair\\output\\clue_counter_arg\\docs"

    grouped:  Dict[str, List[Tuple[str, str]]] = load_all_docs_cleaned(rlp, html_dir)
    tids_score_dict = get_f5_tids_score_d_from_svm()

    def get_score(text):
        if text in tids_score_dict:
            return tids_score_dict[text]
        else:
            return -10000

    class AnalyezedDoc(NamedTuple):
        doc_id: str
        text: str
        score: float
        max_score_sent: str

    for query, entries in grouped.items():
        ad_list = []
        for doc_id, text in entries:
            all_text_list = [text] + sent_tokenize(text)
            scores = lmap(get_score, all_text_list)
            max_idx_ = get_max_idx(scores)
            max_score = scores[max_idx_]
            ad = AnalyezedDoc(doc_id, text, max_score, all_text_list[max_idx_])
            ad_list.append(ad)

        ad_list.sort(key=lambda x: x.score, reverse=True)
        print("QID: ", query)
        for ad in ad_list[:5]:
            rows = [['doc_id', ad.doc_id],
            ['score', ad.score],
            ['max_sent', ad.max_score_sent],
            ['fulltext', ad.text]]
            print("-----")
            print_table(rows)


if __name__ == "__main__":
    main()
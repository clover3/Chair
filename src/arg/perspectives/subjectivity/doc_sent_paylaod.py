import json
import sys
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterator

from nltk.tokenize import sent_tokenize

from arg.mpqa.bert_datagen import encode_w_data_id
from arg.qck.kd_candidate_gen import preload_docs, iterate_docs
from data_generator.tokenizer_wo_tf import get_tokenizer
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
# KD = Knowledge Document
from misc_lib import TimeEstimator, DataIDManager
from tf_util.record_writer_wrap import write_records_w_encode_fn


def sentence_payload_gen(q_res_path: str, top_n, data_id_man: DataIDManager):
    print("loading ranked list")
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    qid_list = list(ranked_list.keys())
    qid_list = qid_list[:10]
    ranked_list = {k: ranked_list[k] for k in qid_list}
    print("Pre loading docs")
    preload_docs(ranked_list, top_n)
    entries: List[Tuple[str, bool, int]] = []

    def enum_sentence(tokens) -> Iterator[str]:
        text = " ".join(tokens)
        sents = sent_tokenize(text)
        yield from sents

    ticker = TimeEstimator(len(ranked_list))
    for qid in ranked_list:
        q_res: List[SimpleRankedListEntry] = ranked_list[qid]
        docs = iterate_docs(q_res, top_n)

        for doc in docs:
            for sent_idx, sent in enumerate(enum_sentence(doc.tokens)):
                info = {
                    'doc_id': doc.doc_id,
                    'sent_idx': sent_idx,
                    'sentence': sent
                }
                data_id = data_id_man.assign(info)
                e = sent, True, data_id
                entries.append(e)

        ticker.tick()
    return entries


def main():
    data_id_man = DataIDManager()
    q_res_path = sys.argv[1]
    save_path = sys.argv[2]
    max_seq_length = 512
    tokenizer = get_tokenizer()
    insts = sentence_payload_gen(q_res_path, 100, data_id_man)

    def encode_fn(t: Tuple[str, bool, int]) -> OrderedDict:
        return encode_w_data_id(tokenizer, max_seq_length, t)

    write_records_w_encode_fn(save_path, encode_fn, insts)
    json_save_path = save_path + ".info"
    json.dump(data_id_man.id_to_info, open(json_save_path, "w"))


if __name__ == "__main__":
    main()
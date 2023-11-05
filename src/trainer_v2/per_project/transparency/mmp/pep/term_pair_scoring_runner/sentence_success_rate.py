import sys
from data_generator.tokenizer_wo_tf import get_tokenizer
from iter_util import load_jsonl
from misc_lib import average, get_second, SuccessCounter
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TokenizedText, get_term_rep
from typing import List, Iterable, Callable, Dict, Tuple, Set


def load_segment_log(jsonl) -> Dict[Tuple[str, str], List[Tuple]]:
    tokenizer = get_tokenizer()
    group_by_pair: Dict[Tuple[str, str], List[Tuple]] = {}
    for j in jsonl:
        if "query" in j:
            query = j['query']
            doc = j['document']
            q_rep: TokenizedText = TokenizedText.from_text(tokenizer, query)
            d_rep: TokenizedText = TokenizedText.from_text(tokenizer, doc)
            group_by_pair[(query, doc)] = list()
        else:
            q_indices = j['q_indices']
            d_indices = j['d_indices']
            score = j["score"]
            q_term = get_term_rep(q_rep, q_indices)
            d_term = get_term_rep(d_rep, d_indices)
            group_by_pair[(query, doc)].append((q_term, d_term, score))
    return group_by_pair


def check(group_by_pair):
    suc = SuccessCounter()
    for (q, d), entries in group_by_pair.items():
        max_score = max([score for q_term, d_term, score in entries])
        if max_score > 1.2:
            suc.suc()
        else:
            print("query: ", q)
            print("document: ", d)
            for q_term, d_term, score in entries:
                print(f"{q_term}/{d_term}/{score:.2f}")

            suc.fail()

    print("Suc rate", suc.get_suc_prob())


def main():
    jsonl = load_jsonl(sys.argv[1])
    group_by_pair = load_segment_log(jsonl)
    check(group_by_pair)


if __name__ == "__main__":
    main()

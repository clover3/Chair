import os.path
from collections import Counter
from typing import List, Iterable, Callable

from adhoc.other.bm25_retriever_helper import get_tokenize_fn2
from misc_lib import get_second, TEL, batch_iter_from_entry_iter, TimeEstimator
from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.misc_common import load_tsv, save_tsv
from trainer_v2.per_project.transparency.mmp.elibm25_baseline.accumulate_q_term import load_expansions


def demo_alignments(expansions, get_dist_batch):
    qt_count = Counter()
    co_count = Counter()
    missing_cnt = 0
    expansions = list(expansions)
    for orig, expanded in TEL(expansions):
        all_pair = set()
        for w in expanded:
            for qt in orig:
                all_pair.add((qt, w))

        all_pair_l = list(all_pair)
        try:
            score_d = dict(zip(all_pair_l, get_dist_batch(all_pair_l)))
            qt_count.update(orig)
            out_row = []
            for w in expanded:
                cands = []
                for qt in orig:
                    try:
                        d = score_d[qt, w]
                        cands.append((qt, d))
                    except KeyError:
                        pass

                cands.sort(key=get_second, reverse=True)

                if cands:
                    qt = cands[0][0]
                    co_count[qt, w] += 1
                    out_row.append(f"{w}->{qt}")

            msg = "{}, {}".format(" ".join(orig), " ".join(out_row))
            print(orig)
            print(expanded)
            print(msg)
            print()
        except KeyError:
            missing_cnt += 1



def main():
    def file_path_iter_all():
        for i in range(120):
            file_path = path_join(output_path, "msmarco", "passage", "eli_q_ex", f"{i}.txt")
            yield file_path

    path_itr = file_path_iter_all()
    tokenize_fn = get_tokenize_fn2("lucene_krovetz")
    expansions = load_expansions(path_itr, tokenize_fn)

    score_save_path = path_join(output_path, "msmarco", "passage", "eli_term_pairs_scores.txt")

    score_d = {}
    for qt, dt, score in load_tsv(score_save_path):
        score_d[qt, dt] = score

    def score_fn(keys):
        return [score_d[k] for k in keys]

    demo_alignments(expansions, score_fn)


if __name__ == "__main__":
    main()


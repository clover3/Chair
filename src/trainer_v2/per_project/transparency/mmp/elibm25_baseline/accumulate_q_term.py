import os.path
from collections import Counter
from typing import List, Iterable, Callable

from adhoc.other.bm25_retriever_helper import get_tokenize_fn2
from misc_lib import get_second, TEL, batch_iter_from_entry_iter, TimeEstimator
from cpath import output_path
from misc_lib import path_join
from ptorch.cross_encoder.get_ce_msmarco_mini_lm import get_ce_msmarco_mini_lm_score_fn
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import load_tsv, save_tsv


def accumulate(expansions: Iterable[tuple[list[str], list[str]]]):
    qt_count = Counter()
    co_count = Counter()
    for orig, expanded in expansions:
        for qt in orig:
            qt_count[qt] += 1
            for w in expanded:
                co_count[qt, w] += 1

    weight_map = avg_distribute(co_count, qt_count)
    return weight_map


def avg_distribute(co_count, qt_count):
    weight_map = {}
    for (qt, w), cnt in co_count.items():
        weight = cnt / qt_count[qt]
        if qt not in weight_map:
            weight_map[qt] = []

        weight_map[qt].append((w, weight))
    return weight_map


def accumulate_w_vector_sim(
        expansions: Iterable[tuple[list[str], list[str]]],
        get_dist_batch: Callable[[list[tuple[str, str]]], list[float]]):
    c_log.info("accumulate_w_vector_sim")
    co_count, qt_count = count_alignments(expansions, get_dist_batch)
    weight_map = avg_distribute(co_count, qt_count)
    return weight_map


def accumulate_min_filter(
        expansions: Iterable[tuple[list[str], list[str]]],
        get_dist_batch: Callable[[list[tuple[str, str]]], list[float]]):
    c_log.info("accumulate_min_filter")
    co_count, qt_count = count_alignments(expansions, get_dist_batch)
    co_count = {(qt, w): cnt for (qt, w), cnt in co_count.items() if qt_count[qt] > 1}
    weight_map = avg_distribute(co_count, qt_count)
    return weight_map


def count_alignments(expansions, get_dist_batch):
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
        except KeyError:
            missing_cnt += 1
    print(f"Missing {missing_cnt} of {len(expansions)}")
    return co_count, qt_count


def compute_vector_sim(
        expansions: Iterable[tuple[list[str], list[str]]],
        get_dist_batch: Callable[[list[tuple[str, str]]], list[float]],
        score_save_path,
):
    c_log.info("accumulate_w_vector_sim")
    save_path = path_join(output_path, "msmarco", "passage", "eli_temr_pairs.txt")
    all_pair_flat = save_term_pairs(expansions, save_path)

    f_out = open(score_save_path, "w")
    n_batch = len(all_pair_flat) // 64
    ticker = TimeEstimator(n_batch)
    for batch in batch_iter_from_entry_iter(all_pair_flat, 64):
        res = get_dist_batch(batch)
        for (qt, dt), score in zip(batch, res):
            f_out.write(f"{qt}\t{dt}\t{score}\n")
        ticker.tick()
    f_out.close()


def save_term_pairs(expansions, save_path):
    expansions = list(expansions)
    all_pair_set = set()
    all_pair_flat = []
    for orig, expanded in TEL(expansions):
        for w in expanded:
            for qt in orig:
                if not (qt, w) in all_pair_set:
                    all_pair_set.add((qt, w))
                    all_pair_flat.append((qt, w))

    print(f"{len(all_pair_flat)} items")

    save_tsv(all_pair_flat, save_path)
    return all_pair_flat


def load_expansions(file_path_iter, tokenize_fn):
    for file_path in file_path_iter:
        if os.path.exists(file_path):
            for row_idx, row in enumerate(load_tsv(file_path)):
                try:
                    qid, expansions, _score, query = row
                    expanded: list[str] = expansions.split()
                    orig: list[str] = tokenize_fn(query)
                    yield orig, expanded
                except ValueError as e:
                    print(e)
                    print(file_path)
                    print(row_idx, row)


def mapping_to_table(mapping: dict[str, list[tuple[str, float]]]) -> Iterable[tuple[str, str, float]]:
    for qt, entries in mapping.items():
        for dt, score in entries:
            yield qt, dt, score


def main():
    def file_path_iter_all():
        for i in range(120):
            file_path = path_join(output_path, "msmarco", "passage", "eli_q_ex", f"{i}.txt")
            yield file_path

    acc_opt = "simple"
    table_name = "eli_D"
    table_save_path = path_join(output_path, "mmp", "tables", f"{table_name}.tsv")
    path_itr = file_path_iter_all()
    tokenize_fn = get_tokenize_fn2("lucene_krovetz")
    expansions = load_expansions(path_itr, tokenize_fn)

    if acc_opt == "simple":
        weight_map = accumulate(expansions)
    else:
        c_log.info("Loading ce model")
        score_fn = get_ce_msmarco_mini_lm_score_fn()
        score_save_path = path_join(output_path, "msmarco", "passage", "eli_term_pairs_scores.txt")
        if not os.path.exists(score_save_path):
            compute_vector_sim(expansions, score_fn, score_save_path)

        score_d = {}
        for qt, dt, score in load_tsv(score_save_path):
            score_d[qt, dt] = score

        def score_fn(keys):
            return [score_d[k] for k in keys]

        weight_map = accumulate_min_filter(expansions, score_fn)

    out_itr = list(mapping_to_table(weight_map))

    def get_key(e):
        return e[0], -e[2]

    table = [e for e in out_itr if e[2] > 0.01]
    table.sort(key=get_key)

    save_tsv(table, table_save_path)
    c_log.info("Saved at %s", table_save_path)


if __name__ == "__main__":
    main()


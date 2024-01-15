from collections import defaultdict
from typing import Dict, List

from table_lib import tsv_iter, tsv_iter_no_quote
from trainer_v2.chair_logging import c_log


def load_mapping_from_align_scores(
        tsv_path, cut, mapping_val) -> Dict[str, Dict[str, float]]:
    rows = tsv_iter(tsv_path)

    n_entry = 0
    mapping = defaultdict(dict)
    for q_term, d_term, score in rows:
        if float(score) > cut:
            mapping[q_term][d_term] = mapping_val
            n_entry += 1

    c_log.info("%d entry loaded", n_entry)
    return mapping


def load_align_scores(tsv_path) -> Dict[str, Dict[str, float]]:
    if tsv_path.lower() == "none":
        rows = []
    else:
        rows = tsv_iter(tsv_path)

    n_entry = 0
    mapping: Dict[str, Dict[str, float]] = defaultdict(dict)
    for q_term, d_term, score in rows:
        mapping[q_term][d_term] = float(score)
        n_entry += 1

    c_log.info("%d entry loaded", n_entry)
    return mapping


def load_mapping_from_align_candidate(
        tsv_path, mapping_val) -> Dict[str, Dict[str, float]]:
    rows = tsv_iter(tsv_path)

    n_entry = 0
    mapping = defaultdict(dict)
    for q_term, d_term in rows:
        mapping[q_term][d_term] = mapping_val
        n_entry += 1

    c_log.info("%d entry loaded", n_entry)
    return mapping


def load_binary_mapping_from_align_candidate(tsv_path) -> Dict[str, List[str]]:
    c_log.debug("load_binary_mapping_from_align_candidate")

    rows = tsv_iter_no_quote(tsv_path)

    n_entry = 0
    mapping = defaultdict(list)
    for row in rows:
        q_term, d_term = row
        mapping[q_term].append(d_term)
        n_entry += 1
        assert "\n" not in q_term
        assert "\n" not in d_term

    c_log.info("%d entry loaded", n_entry)
    return mapping


def load_binary_mapping_from_align_scores(
        tsv_path, cut) -> Dict[str, List[str]]:
    rows = tsv_iter_no_quote(tsv_path)

    n_entry = 0
    mapping = defaultdict(list)
    for row in rows:
        if len(row) == 3:
            q_term, d_term, score = row
            if float(score) > cut:
                mapping[q_term].append(d_term)
                n_entry += 1
        else:
            q_term, d_term = row
            mapping[q_term].append(d_term)
            n_entry += 1

    c_log.info("%d entry loaded", n_entry)
    return mapping

import csv
import os
from typing import NamedTuple

import scipy.stats

from cpath import output_path
from list_lib import lmap, lfilter


class Entry(NamedTuple):
    avg_value: float
    doc_id: str
    part_idx: int
    predicted_score: float
    cid: int


def read():
    file_path = os.path.join(output_path, "visualize", "doc_value.tsv")
    cur_cid = -1
    for row in csv.reader(open(file_path, "r", encoding="utf-8"), delimiter="\t"):
        try:
            avg_value = float(row[0])
            doc_id = row[1]
            part_idx = int(row[2])
            predicted_score = float(row[3])
            e = Entry(avg_value, doc_id, part_idx, predicted_score, cur_cid)
            yield e
        except ValueError as e:
            try:
                a, b = row[0].split(":")
                cid = int(a)
                cur_cid = cid
            except:
                pass
            pass


def get_correlation(series_a, series_b):
    return scipy.stats.pearsonr(series_a, series_b)


def cases():
    entries = list(read())
    good_doc = set()
    good_prediction = set()

    for e in entries:
        if e.predicted_score > 0.9:
            good_doc.add(e.doc_id)
            if e.avg_value > 0.01:
                good_prediction.add(e.doc_id)

    for e in entries:
        if e.doc_id in good_prediction:
            print(e.doc_id, e.part_idx, e.avg_value, e.predicted_score, )


def stats2():
    entries = list(read())
    print("Total items", len(entries))

    def is_good(x):
        return x.avg_value > 0.01
    good_doc = set()
    good_cid = set()
    cid_first_good_doc = dict()
    for e in entries:
        if is_good(e):
            good_doc.add(e.doc_id)
            good_cid.add(e.cid)
            cid_first_good_doc[e.cid] = e.doc_id

    good_doc_entries = lfilter(lambda x: x.doc_id in good_doc, entries)
    good_cid_entries = lfilter(lambda x: x.cid in good_cid, entries)
    good_cid_other_doc_entries = lfilter(lambda x: x.cid in good_cid and x.doc_id not in cid_first_good_doc[x.cid], entries)
    good_in_good_doc = lfilter(is_good, good_doc_entries)
    good_in_good_cid = lfilter(is_good, good_cid_entries)
    good_in_good_cid_other_doc = lfilter(is_good, good_cid_other_doc_entries)

    n_good_in_good_doc = len(good_in_good_doc)
    print("Good rate in good docs", get_rate_str(n_good_in_good_doc, len(good_doc_entries)))
    n_good_in_good_doc_adj = n_good_in_good_doc - len(good_doc)
    nom = len(good_doc_entries) - len(good_doc)
    print("Good rate* in remaining good docs", get_rate_str(n_good_in_good_doc_adj, nom))

    denom = len(good_in_good_cid) - len(good_cid)
    nom = len(good_cid_entries) - len(good_cid)
    print("Good rate* in remaining good cids", get_rate_str(denom, nom))

    denom = len(good_in_good_cid_other_doc) - len(good_cid)
    nom = len(good_cid_other_doc_entries) - len(good_cid)
    print("Good rate* in good cids, excluding first good doc", get_rate_str(denom, nom))


def get_rate_str(a, b):
    return "{0:.4f} = {1}/{2}".format(a / b, a, b)


def stats():
    entries = list(read())
    print("Total items", len(entries))

    unique = set()
    for e in entries:
        unique.add((e.doc_id, e.part_idx))

    print("Unique passages", len(unique))

    avg_value_list = lmap(lambda x: x.avg_value, entries)
    predicted_score_list = lmap(lambda x: x.predicted_score, entries)

    good_doc = set()
    for e in entries:
        if e.predicted_score > 0.9:
            good_doc.add(e.doc_id)

    r = get_correlation(avg_value_list, predicted_score_list)
    print(r)

    over_09 = lfilter(lambda x: x.predicted_score > 0.9, entries)
    under_01 = lfilter(lambda x: x.predicted_score < 0.1, entries)
    doc_over_09 = lfilter(lambda x: x.doc_id in good_doc, entries)
    doc_over_09_under_01 = lfilter(lambda x: x.doc_id in good_doc, under_01)

    def is_good(x):
        return x.avg_value > 0.01

    def is_bad(x):
        return x.avg_value < -0.01

    for criteria in [is_good, is_bad]:
        good_global = lfilter(criteria, entries)
        good_over_09 = lfilter(criteria, over_09)
        good_under_01 = lfilter(criteria, under_01)
        good_doc_over_09 = lfilter(criteria, doc_over_09)
        good_doc_over_09_under_01 = lfilter(criteria, doc_over_09_under_01 )

        job = criteria.__name__
        print("global {} rate".format(job), get_rate_str(len(good_global), len(entries)))
        print("over 09 {} rate".format(job), get_rate_str(len(good_over_09), len(over_09)))
        print("under 01 {} rate".format(job), get_rate_str(len(good_under_01), len(under_01)))
        print("doc over 09 {} rate".format(job), get_rate_str(len(good_doc_over_09), len(doc_over_09)))
        print("doc over 09 under 01 {} rate".format(job), get_rate_str(len(good_doc_over_09_under_01), len(doc_over_09_under_01)))


if __name__ == "__main__":
    stats()

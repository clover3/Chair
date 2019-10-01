from crs.load_stance_annotation import load_stance_annot
from path import output_path
import os
from elastic.retrieve import get_comment, get_paragraph, get_title
import csv


def generate(path):
    r = load_stance_annot(path)

    out = csv.writer(open("output.csv", "w"))

    veri_list = []

    hash_set = set()
    for e in r:
        statement =  e['statement']

        if e['support'] >= 1 and len(e['support_evidence']) > 1:
            doc_id, seg_id = e['support_evidence']
            sig = statement + doc_id + seg_id
            if sig not in hash_set:
                hash_set.add(sig)
                veri_list.append((statement, doc_id, seg_id))

        if e['dispute'] >= 1 and len(e['dispute_evidence']) > 1:
            doc_id, seg_id = e['dispute_evidence']
            sig = statement + doc_id + seg_id
            if sig not in hash_set:
                hash_set.add(sig)
                veri_list.append((statement, doc_id, seg_id))

    out.writerows(veri_list)

def gen1():
    path = "C:\work\Data\CKB annotation\dipsute annotation 1\\Batch_3746208_batch_results.csv"
    generate(path)


gen1()

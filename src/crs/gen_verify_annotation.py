from crs.load_stance_annotation import load_stance_annot
from path import output_path
import os
from elastic.retrieve import get_comment, get_paragraph, get_title
import csv


def generate(path_list):
    r = []
    for path in path_list:
        r.extend(load_stance_annot(path))

    out = csv.writer(open("output.csv", "w"), lineterminator="\n")

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

    veri_list_formed = []
    for statement, doc_id, seg_id in veri_list:
        link = "http://gosford.cs.umass.edu/search/#!/view3/{}/{}".format(seg_id.strip(), doc_id.replace("/", "_"))
        veri_list_formed.append((statement, link))


    column = []
    for i in range(1,6):
        column.append("statement{}".format(i))
        column.append("link{}".format(i))

    folded_list = [column]
    for idx in range(0, len(veri_list_formed), 5):

        line = []
        for i in range(5):
            j = idx + i
            if j >= len(veri_list_formed):
                break
            statement, link = veri_list_formed[j]
            line.append(statement)
            line.append(link)
        if len(line) == 2*5:
            folded_list.append(line)

    print(len(folded_list))
    out.writerows(folded_list)


def gen1():
    path1 = "C:\work\Data\CKB annotation\Search Stances 4\\Batch_3749275_batch_results.csv"
    path2 = "C:\work\Data\CKB annotation\Search Stances 5\\Batch_3779961_batch_results.csv"
    generate([path1, path2])


gen1()

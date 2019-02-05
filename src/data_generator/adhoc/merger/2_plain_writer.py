import os
from data_generator.data_parser.trec import load_trec, load_robust
import pickle
import sys
import csv

def load_marco_query(path):
    f = open(path, "r")
    queries = dict()
    for row in csv.reader(f, delimiter="\t"):
        q_id = row[0]
        query = row[1]
        queries[q_id] = query
    return queries


def split_text(content):
    idx = 0
    window_size = 200 * 3
    while idx < len(content):
        span = content[idx:idx + window_size]
        yield span
        idx += window_size


def pickle_payload(slave_id):
    query_path = "/mnt/nfs/work3/youngwookim/code/adhoc/robus/queries.train.tsv"
    docs = load_robust("/mnt/nfs/work3/youngwookim/data/robust04")
    query = load_marco_query(query_path)

    print("{}] Load sampled".format(slave_id))
    dir_path = "/mnt/nfs/work3/youngwookim/code/adhoc/robus/marco_query_doc_results"

    sample_filename = os.path.join(dir_path, "merger_train_{}.pickle".format(slave_id))
    sampled = pickle.load(open(sample_filename, "rb"))

    inst = []
    for q_id_1, doc_id_1, q_id_2, doc_id_2 in sampled:
        q1 = query[q_id_1]
        runs_1 = []
        for span in split_text(docs[doc_id_1]):
            runs_1.append((q1, span))

        q2 = query[q_id_2]
        runs_2 = []
        for span in split_text(docs[doc_id_2]):
            runs_2.append((q2, span))

        inst.append((runs_1, runs_2))
    print(len(inst))
    for i in range(10):
        st = i * 1000
        ed = (i+1) * 1000
        name = str(slave_id) + str(i)
        pickle.dump(inst[st:ed], open("../output/merger_plainpair_{}.pickle".format(name), "wb"))



if __name__ == "__main__":
    pickle_payload(int(sys.argv[1]))

import os
from data_generator.data_parser.trec import load_trec, load_robust
import csv
import pickle
import sys

def load_marco_query(path):
    f = open(path, "r")
    queries = dict()
    for row in csv.reader(f, delimiter="\t"):
        q_id = row[0]
        query = row[1]
        queries[q_id] = query
    return queries


def pickle_payload(slave_id):
    query_path = "/mnt/nfs/work3/youngwookim/code/adhoc/robus/queries.train.tsv"
    docs = load_robust("/mnt/nfs/work3/youngwookim/data/robust04")
    query = load_marco_query(query_path)

    print("{}] Load sampled".format(slave_id))

    sample_filename =  "id_pair_{}.pickle".format(slave_id)
    sampled = pickle.load(open(sample_filename, "rb"))

    inst = []
    for q_id_1, doc_id_1, q_id_2, doc_id_2 in sampled:
        q1 = query[q_id_1]
        d1 = docs[doc_id_1]

        q2 = query[q_id_2]
        d2 = docs[doc_id_2]
        inst.append((q1, d1, q2, d2))

    print(len(inst))
    step = 1000
    n_block = int(len(inst) / step)
    for i in range(n_block):
        st = i * step
        ed = (i+1) * step
        name = str(slave_id) + "_" + str(i)
        pickle.dump(inst[st:ed], open("../output/plain_pair_{}.pickle".format(name), "wb"))


if __name__ == "__main__":
    pickle_payload(int(sys.argv[1]))

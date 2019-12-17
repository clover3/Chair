import os
import pickle

from data_generator import tokenizer_wo_tf as tokenization
from misc_lib import TimeEstimator
from path import data_path
from sydney_manager import MTM
from tlm.wiki.generate_bert_tf import parse_wiki

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"

def work(job_id, all_docs, tokenizer):
    j = job_id % 100
    interval = int(len(all_docs) / 100) + 1
    st = j * interval
    ed = st + interval

    out_docs = []
    # Empty lines are used as document delimiters
    docs = all_docs[st:ed]
    ticker = TimeEstimator(len(docs))
    for doc in docs:
        out_docs.append([])
        for line in doc:
            line = line.strip()
            tokens = tokenizer.tokenize(line)
            if tokens:
                out_docs[-1].append(tokens)

        ticker.tick()

    out_pickle = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_train_tokens.{}".format(job_id)
    pickle.dump(out_docs, open(out_pickle, "wb"))


def main():
    mark_path = os.path.join(working_path, "wiki_token")
    out_path = os.path.join(working_path, "tf")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    mtm = MTM(1000, mark_path)
    vocab_file = os.path.join(data_path, "bert_voca.txt")

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    docs_dict = {}
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        i = int(job_id / 100)
        if i not in docs_dict:
            file_path = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_train.txt.line.{}".format(i)
            docs_dict[i] = parse_wiki(file_path)
        work(job_id, docs_dict[i], tokenizer)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)


if __name__ == "__main__":
    main()


import os
import pickle

from data_generator import tokenizer_wo_tf as tokenization
from path import data_path
from sydney_manager import MTM
from tlm.wiki.generate_bert_tf import parse_wiki

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"


def work(job_id, all_docs, tokenizer, out_path_format):
    j = job_id % 100
    interval = int(len(all_docs) / 100) + 1
    st = j * interval
    ed = st + interval

    out_docs = []
    # Empty lines are used as document delimiters
    docs = all_docs[st:ed]
    for doc in docs:
        out_docs.append([])
        for line in doc:
            line = line.strip()
            tokens = tokenizer.tokenize(line)
            if tokens:
                out_docs[-1].append(tokens)


    out_pickle = out_path_format.format(job_id)
    pickle.dump(out_docs, open(out_pickle, "wb"))


def main():
    mark_path = os.path.join(working_path, "wiki_eval_token")
    mtm = MTM(100, mark_path)
    vocab_file = os.path.join(data_path, "bert_voca.txt")

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    docs_dict = {}
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    todo = "dev"
    if todo == "train":
        out_path_format = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_train_tokens.{}"
        in_path_format = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_train.txt.line.{}"
    elif todo == "dev":
        out_path_format = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_eval_tokens.{}"
        in_path_format = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_eval.txt.line.{}"
    else:
        assert False

    while job_id is not None:
        i = int(job_id / 100)
        if i not in docs_dict:
            file_path = in_path_format.format(i)
            docs_dict[i] = parse_wiki(file_path)
        work(job_id, docs_dict[i], tokenizer, out_path_format)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)


if __name__ == "__main__":
    main()


import os
import random

from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_encoders import AlignCandidateExtractor
import sys
import sys
from omegaconf import OmegaConf
from table_lib import tsv_iter
import itertools

from cpath import output_path
from misc_lib import path_join, batch_iter_from_entry_iter


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    bert_tokenizer = get_tokenizer()
    extractor = AlignCandidateExtractor(bert_tokenizer.basic_tokenizer.tokenize)
    line_per_job = conf.line_per_job
    raw_train_iter = tsv_iter(conf.qd_triplet_file)

    random.seed(0)
    start_job = 291
    st = line_per_job * start_job
    ed = st + line_per_job
    itr = itertools.islice(raw_train_iter, st, ed)
    print("islice")
    for item in itr:
        print(item)
        break

    raw_train_iter = tsv_iter(conf.qd_triplet_file)
    itr = itertools.islice(raw_train_iter, st, None)
    itr = batch_iter_from_entry_iter(itr, line_per_job)
    for idx, batch in enumerate(itr):
        for item in batch:
            print(item)
            break
        break


if __name__ == "__main__":
    main()
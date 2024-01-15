import json
import math
import random
from typing import Iterable

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.passage.path_helper import get_train_triples_partition_path
from misc_lib import pick1, ceil_divide, path_join, TimeEstimator
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import read_lines
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TextRep
import sys
from omegaconf import OmegaConf


def write_lookup_todo(
        job_no,
        qdd_itr: Iterable[tuple[str, str, str]],
        save_dir,
        voca,
        voca_d,
        n_item):
    tokenizer = get_tokenizer()
    term_for_worker = 1000
    n_worker = ceil_divide(len(voca), term_for_worker)
    f_arr = []
    for i in range(n_worker):
        file_name = f"{job_no}_{i}"
        f = open(path_join(save_dir, file_name), "w")
        f_arr.append(f)

    def get_out_f(q_term):
        term_idx = voca_d[q_term]
        worker_i = term_idx // term_for_worker
        return f_arr[worker_i]

    def get_match_candidate(q, d):
        q = TextRep.from_text(tokenizer, q)
        d: TextRep = TextRep.from_text(tokenizer, d)

        d_term_list = list(d.counter.keys())
        q_term_cand = []
        for q_term, qtf, _ in q.get_bow():
            exact_match_cnt: int = d.counter[q_term]
            if not exact_match_cnt:
                q_term_cand.append(q_term)

        q_term_cand = [t for t in q_term_cand if t in voca_d]
        d_term_list = [t for t in d_term_list if t in voca_d]
        q_term = pick1(q_term_cand) if q_term_cand else None
        return q_term, d_term_list

    ticker = TimeEstimator(n_item)
    for idx, (q, d_pos, d_neg) in enumerate(qdd_itr):
        data_id_pos = idx * 10 + 1
        data_id_neg = idx * 10 + 2
        for q, d, data_id in [(q, d_pos, data_id_pos), (q, d_neg, data_id_neg)]:
            q_term, d_term_list = get_match_candidate(q, d)
            if q_term is not None and d_term_list:
                q_term_id = voca_d[q_term]
                d_term_ids = [voca_d[t] for t in d_term_list]
                out_row = [data_id, q_term_id, d_term_ids]
                out_f = get_out_f(q_term)
                s = json.dumps(out_row)
                out_f.write(s + "\n")
            ticker.tick()


# confs/experiment_confs/datagen_pep_tt7.yaml
def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    job_no = sys.argv[2]
    file_path = get_train_triples_partition_path(job_no)
    random.seed(0)
    raw_train_iter: Iterable[tuple[str, str, str]] = tsv_iter(file_path)
    voca: list[str] = read_lines(conf.voca_path)
    voca_d = {t: idx for idx, t in enumerate(voca)}
    save_dir: str = conf.lookup_todo_dir
    write_lookup_todo(job_no, raw_train_iter, save_dir, voca, voca_d, 1000000)


if __name__ == "__main__":
    main()





import itertools
import os
import sys
from collections import OrderedDict

from dataset_specific.msmarco.passage.path_helper import get_train_triples_partition_path
from list_lib import left
from misc_lib import path_join, batch_iter_from_entry_iter, get_second
from omegaconf import OmegaConf
from cpath import output_path, at_output_dir
from table_lib import tsv_iter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import get_pep_tt_single_encoder, load_table, get_pep_tt_single_encoder_for_with_align_info
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


def iter_partitions_and_info(
        align_info_conf, partition_idx_st, partition_idx_ed) -> Iterable[tuple[list, dict]]:
    partition_size = align_info_conf.line_per_job
    raw_train_iter = tsv_iter(align_info_conf.qd_triplet_file)

    line_st = partition_idx_st * partition_size
    line_ed = partition_idx_ed * partition_size

    line_itr = itertools.islice(raw_train_iter, line_st, line_ed)
    batch_itr = batch_iter_from_entry_iter(line_itr, partition_size)
    for idx, partition in enumerate(batch_itr):
        partition_idx = idx + partition_idx_st
        table_path = path_join(align_info_conf.score_save_dir, f"{partition_idx}.txt")
        if not os.path.exists(table_path) or os.path.getsize(table_path) == 0:
            c_log.warn("File does not exists skip this partition %s", table_path)
            continue

        output_d = load_table(table_path)
        yield partition, output_d


def run_pep_tt_encoding_jobs(encoder, conf, job_no):
    align_info_conf = OmegaConf.load(conf.align_info_conf)

    n_partition_per_job = 100
    n_item_per_job = n_partition_per_job * align_info_conf.line_per_job
    partition_idx_st = job_no * n_partition_per_job
    partition_idx_ed = partition_idx_st + n_partition_per_job

    def encode_partition(encoder, partition, output_d) -> Iterable[OrderedDict]:
        def get_pep_top_k(q_term, d_term_iter) -> List[str]:
            score_d_per_q_term = output_d[q_term]
            d_term_list = list(d_term_iter)
            scores = []
            for d_term in d_term_list:
                try:
                    scores.append((d_term, score_d_per_q_term[d_term]))
                except KeyError:
                    pass

            scores.sort(key=get_second, reverse=True)
            return left(scores)

        encoder.bm25_analyzer.get_pep_top_k = get_pep_top_k
        for row in partition:
            q = row[0]
            d_pos = row[1]
            d_neg = row[2]
            feature_d = encoder.encode_triplet(q, d_pos, d_neg)
            yield feature_d

    save_path = path_join(conf.tfrecord_save_dir, str(job_no))

    def encode_fn(d) -> OrderedDict:
        return encoder.to_tf_feature(d)

    def feature_itr():
        for partition, output_d in iter_partitions_and_info(align_info_conf, partition_idx_st, partition_idx_ed):
            yield from encode_partition(encoder, partition, output_d)

    write_records_w_encode_fn(save_path, encode_fn, feature_itr(), n_item_per_job)



def main():
    conf = OmegaConf.load(sys.argv[1])
    model_config = PEP_TT_ModelConfig()
    job_no = int(sys.argv[2])
    c_log.info("Job %d", job_no)
    encoder = get_pep_tt_single_encoder_for_with_align_info(model_config, conf)
    run_pep_tt_encoding_jobs(encoder, conf, job_no)


if __name__ == "__main__":
    main()

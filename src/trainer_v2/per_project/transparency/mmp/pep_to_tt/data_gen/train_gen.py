import itertools
import sys
from collections import OrderedDict

from dataset_specific.msmarco.passage.path_helper import get_train_triples_partition_path
from misc_lib import path_join
from omegaconf import OmegaConf
from cpath import output_path, at_output_dir
from table_lib import tsv_iter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import get_pep_tt_single_encoder, \
    PEP_TT_DatasetBuilder
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


def main():
    conf = OmegaConf.load(sys.argv[1])
    model_config = PEP_TT_ModelConfig()
    encoder = get_pep_tt_single_encoder(model_config, conf)

    data_name = "pep_tt2"
    job_no = int(sys.argv[2])

    file_no = job_no // 10
    st = job_no % (1000 * 1000)
    n_item_per_job = 100 * 1000
    ed = st + n_item_per_job

    file_path = get_train_triples_partition_path(file_no)
    save_dir = at_output_dir("tfrecord", data_name)
    save_path = path_join(save_dir, str(job_no))

    raw_train_iter: Iterable[tuple[str, str, str]] = tsv_iter(file_path)
    raw_train_iter = itertools.islice(raw_train_iter, st, ed)

    def encode_fn(qdd) -> OrderedDict:
        d = encoder.encode_triplet(*qdd)
        return encoder.to_tf_feature(d)

    write_records_w_encode_fn(save_path, encode_fn, raw_train_iter, n_item_per_job)


if __name__ == "__main__":
    main()

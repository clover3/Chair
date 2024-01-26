import sys
from collections import OrderedDict
from typing import Iterable

from omegaconf import OmegaConf

from cpath import at_output_dir
from dataset_specific.msmarco.passage.dev1000_B import iter_dev_split_sample_pairwise
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import get_pep_tt_single_encoder
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


def main():
    conf = OmegaConf.load(sys.argv[1])
    model_config = PEP_TT_ModelConfig()
    encoder = get_pep_tt_single_encoder(model_config, conf)

    data_name = "pep_tt2_val"
    raw_train_iter: Iterable[tuple[str, str, str]] = iter_dev_split_sample_pairwise("dev_sample1000_B")
    save_path = at_output_dir("tfrecord", data_name)

    def encode_fn(qdd) -> OrderedDict:
        d = encoder.encode_triplet(*qdd)
        return encoder.to_tf_feature(d)

    n_item_per_job = 1000
    write_records_w_encode_fn(save_path, encode_fn, raw_train_iter, n_item_per_job)


if __name__ == "__main__":
    main()

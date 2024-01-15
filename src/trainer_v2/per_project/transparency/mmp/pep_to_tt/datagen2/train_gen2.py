import sys
from typing import Iterable
from typing import Iterator

from omegaconf import OmegaConf

from cpath import at_output_dir
from dataset_specific.msmarco.passage.path_helper import get_train_triples_partition_path
from misc_lib import path_join
from table_lib import tsv_iter
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.misc_common import read_lines
from trainer_v2.per_project.transparency.mmp.pep_to_tt.datagen2.align_info_encoder import AlignInfo, TripletAlign, \
    PEP_TT_Encoder7, join_triplet_with_align, parse_align_info, json_iterator
from trainer_v2.per_project.transparency.mmp.pep_to_tt.datagen2.merge_generators import merge_sorted_generators
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig



def load_align_info(conf, part_no) -> Iterator[AlignInfo]:
    dir_path = conf.lookup_output_dir
    voca: list[str] = read_lines(conf.voca_path)
    voca_d: dict[int, str] = {idx: t for idx, t in enumerate(voca)}

    def parse(item) -> AlignInfo:
        return parse_align_info(item, voca_d)

    n_worker = 100
    generator_list = []
    for i in range(n_worker):
        file_name = f"{part_no}_{i}"
        file_path = path_join(dir_path, file_name)
        json_itr = json_iterator(file_path)
        generator_list.append(map(parse, json_itr))

    return merge_sorted_generators(generator_list)


def main():
    conf = OmegaConf.load(sys.argv[1])
    model_config = PEP_TT_ModelConfig()
    encoder = PEP_TT_Encoder7(model_config, conf)

    data_name = "pep_tt7"
    job_no = int(sys.argv[2])

    n_item_per_job = 1000000
    save_dir = at_output_dir("tfrecord", data_name)
    save_path = path_join(save_dir, str(job_no))

    file_path = get_train_triples_partition_path(job_no)
    raw_train_iter: Iterable[tuple[str, str, str]] = tsv_iter(file_path)
    align_info: Iterator[AlignInfo] = load_align_info(conf, job_no)

    itr: Iterator[TripletAlign] = join_triplet_with_align(align_info, raw_train_iter)
    write_records_w_encode_fn(save_path, encoder.encode, itr, n_item_per_job)


if __name__ == "__main__":
    main()

import random
import sys
from typing import Iterable

from omegaconf import OmegaConf

from dataset_specific.msmarco.passage.dev1000_B import iter_dev_split_sample_pairwise
from trainer_v2.per_project.transparency.misc_common import read_lines
from trainer_v2.per_project.transparency.mmp.pep_to_tt.datagen2.write_lookup_todo import write_lookup_todo


# confs/experiment_confs/datagen_pep_tt7.yaml
def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    random.seed(0)
    qdd_iter: Iterable[tuple[str, str, str]] = iter_dev_split_sample_pairwise("dev_sample1000_B")
    voca: list[str] = read_lines(conf.voca_path)
    voca_d = {t: idx for idx, t in enumerate(voca)}
    save_dir: str = conf.val_lookup_todo_dir
    write_lookup_todo(0, qdd_iter, save_dir, voca, voca_d, 1000)


if __name__ == "__main__":
    main()

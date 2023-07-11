import logging
import pickle
import sys
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignConfig, \
    MMPGAlignPathHelper, get_mmp_galign_path_helper


def main():
    config: MMPGAlignPathHelper = get_mmp_galign_path_helper()
    split = "train"
    i = int(sys.argv[1])
    count, rel_count = config.load_tf_occur(split, i)
    save_path = config.get_sub_dir_partition_path("rel_count", str(i))
    pickle.dump(rel_count, open(save_path, "wb"))


if __name__ == "__main__":
    main()

from cpath import yconfig_dir_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignPathHelper, \
    get_mmp_galign_path_helper


#
# def load_path_config() -> MMPGAlignPathHelper:
#     config_path = path_join(yconfig_dir_path, "mmp_galign3.yaml")
#     conf = OmegaConf.structured(MMPGAlignConfig)
#     conf.merge_with(OmegaConf.load(config_path))
#     return MMPGAlignPathHelper(conf)
#

def main():
    config: MMPGAlignPathHelper = get_mmp_galign_path_helper()
    partition_no = 1
    split = "train"
    pickle_save_path = config.get_tf_occur_save_path(split, partition_no)
    print(pickle_save_path)
    open(pickle_save_path, "w").write("s")


if __name__ == "__main__":
    main()

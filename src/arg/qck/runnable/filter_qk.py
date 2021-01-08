from arg.qck.filter_qk_w_ranked_list import filter_with_ranked_list_path
from cache import save_to_pickle
from exec_lib import run_func_with_config


def main(config):
    new_qks = filter_with_ranked_list_path(config['qk_name'],
                                           config['ranked_list_path'],
                                           config['threshold'],
                                           config['top_k'])

    save_to_pickle(new_qks, config['save_name'])


if __name__ == "__main__":
    run_func_with_config(main)

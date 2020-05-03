from explain.runner.train_ex import train_self_explain
from tf_util.tf_logging import tf_logging


def get_params(start_model_path, start_type, info_fn_name, num_gpu):
    return NotImplemented


def train_from(start_model_path, start_type, save_dir,
               modeling_option, tags, info_fn_name, num_deletion,
               g_val=0.5,
               drop_thres=0.3,
               num_gpu=1):

    num_deletion = int(num_deletion)
    num_gpu = int(num_gpu)
    tf_logging.info("train_from : msmarco_ex")
    data, data_loader, hp, informative_fn, init_fn, train_config\
        = get_params(start_model_path, start_type, info_fn_name, num_gpu)

    train_config.num_deletion = num_deletion
    train_config.g_val = float(g_val)
    train_config.drop_thres = float(drop_thres)

    train_self_explain(hp, train_config, save_dir,
                       data, data_loader, tags, modeling_option,
                       init_fn, informative_fn)


from explain.ex_train_modules import action_penalty
from explain.msmarco import Hyperparam, ExTrainConfig, load_data
from explain.runner.train_ex import train_self_explain
from explain.setups import init_fn_generic
from tf_util.tf_logging import tf_logging


def tag_informative(tag, before_prob, after_prob, action):
    score = before_prob[1] - after_prob[1]
    score = score - action_penalty(action)
    return score


def get_params(start_model_path, start_type, num_gpu):
    hp = Hyperparam()
    data = load_data(hp.batch_size)
    # Data : Tuple[train batches, dev batches]
    data_loader = None
    train_config = ExTrainConfig()
    train_config.num_gpu = num_gpu

    def init_fn(sess):
        return init_fn_generic(sess, start_type, start_model_path)

    return data, data_loader, hp, tag_informative, init_fn, train_config


def train_from(start_model_path,
               start_type,
               save_dir,
               modeling_option,
               num_deletion,
               g_val=0.5,
               num_gpu=1,
               drop_thres=0.3):

    num_deletion = int(num_deletion)
    num_gpu = int(num_gpu)
    tf_logging.info("train_from : msmarco_ex")
    data, data_loader, hp, informative_fn, init_fn, train_config\
        = get_params(start_model_path, start_type, num_gpu)

    tags = ["relevant"]
    train_config.num_deletion = num_deletion
    train_config.g_val = float(g_val)
    train_config.drop_thres = float(drop_thres)

    train_self_explain(hp, train_config, save_dir,
                       data, data_loader, tags, modeling_option,
                       init_fn, informative_fn)


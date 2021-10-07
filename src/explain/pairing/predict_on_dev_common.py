from data_generator.shared_setting import BertNLI
from explain.pairing.probe_train_common import NLIPairingTrainConfig
from explain.pairing.runner.run_train import HPCommon
from explain.train_nli import get_nli_data
from tf_util.tf_logging import set_level_debug, reset_root_log_handler


def prepare_predict_setup(num_gpu):
    num_gpu = int(num_gpu)
    hp = HPCommon()
    nli_setting = BertNLI()
    set_level_debug()
    reset_root_log_handler()
    train_config = NLIPairingTrainConfig()
    train_config.num_gpu = num_gpu
    data = get_nli_data(hp, nli_setting)
    train_batches, dev_batches = data
    return dev_batches, hp, train_config
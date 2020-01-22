import sys

from data_generator.NLI.nli import get_modified_data_loader
from data_generator.common import get_tokenizer
from data_generator.shared_setting import BertNLI
from explain.runner.train_ex import train_nli_ex
from explain.train_nli import get_nli_data
from models.transformer import hyperparams
from tf_util.tf_logging import tf_logging, set_level_debug


def train_one_tag(start_model_path, save_dir, modeling_option, tag):
    tf_logging.info("train_one_tag : nli_ex")
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    max_steps = 73630
    set_level_debug()

    tokenizer = get_tokenizer()
    tf_logging.info("Intializing dataloader")
    data_loader = get_modified_data_loader(tokenizer, hp.seq_max, nli_setting.vocab_filename)
    tf_logging.info("loading batches")
    data = get_nli_data(hp, nli_setting)
    tags = [tag]
    train_nli_ex(hp, nli_setting, save_dir, max_steps, data, data_loader, start_model_path, tags, modeling_option)


if __name__  == "__main__":
    train_one_tag(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
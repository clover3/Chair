import sys

from cache import load_cache, save_to_pickle
from data_generator.NLI import nli
from data_generator.shared_setting import BertNLI
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.train_nli import train_nli_multi_gpu
from models.transformer import hyperparams
from tf_util.tf_logging import set_level_debug
from trainer.model_saver import load_model_w_scope, tf_logger
from trainer.tf_module import get_nli_batches_from_data_loader


def get_snli_data(hp, nli_setting):
    data_loader = nli.SNLIDataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    tokenizer = get_tokenizer()
    CLS_ID = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    data_loader.CLS_ID = CLS_ID
    data_loader.SEP_ID = SEP_ID
    cache_name = "snli_batch{}_seq{}".format(hp.batch_size, hp.seq_max)
    data = load_cache(cache_name)
    if data is None:
        tf_logger.info("Encoding data from csv")
        data = get_nli_batches_from_data_loader(data_loader, hp.batch_size)
        save_to_pickle(data, cache_name)
    return data


def train_nil_from_bert(model_path, save_dir):
    def load_fn(sess, model_path):
        return load_model_w_scope(sess, model_path, "bert")
    max_steps = 61358
    max_steps = 36250
    hp = hyperparams.HPSENLI3()
    set_level_debug()
    nli_setting = BertNLI()
    data = get_snli_data(hp, nli_setting)
    n_gpu = 2
    return train_nli_multi_gpu(hp, nli_setting, save_dir, max_steps, data, model_path, load_fn, n_gpu)


if __name__  == "__main__":
    train_nil_from_bert(sys.argv[1], sys.argv[2])
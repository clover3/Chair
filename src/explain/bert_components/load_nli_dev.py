from data_generator.NLI.nli import get_modified_data_loader
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.bert_components.cmd_nli import ModelConfig
from trainer.np_modules import get_batches_ex


def load_data(model_config: ModelConfig):
    vocab_filename = "bert_voca.txt"
    data_loader = get_modified_data_loader(get_tokenizer(), model_config.max_seq_length, vocab_filename)
    dev_insts = data_loader.get_dev_data()
    return get_batches_ex(dev_insts, 128, 4)
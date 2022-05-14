from trainer_v2.run_config import RunConfigEx


class ModelConfig:
    num_classes = 3
    max_seq_length = 512

    def __init__(self, bert_config, max_seq_length):
        self.bert_config = bert_config
        self.max_seq_length = max_seq_length


class MultiSegModelConfig:
    num_classes = 3
    def __init__(self, bert_config, max_seq_length_list):
        self.bert_config = bert_config
        self.max_seq_length_list = max_seq_length_list


def get_run_config_nli_train(args):
    steps_per_epoch = 25000
    num_epochs = 4
    run_config = RunConfigEx(model_save_path=args.output_dir,
                             train_step=num_epochs * steps_per_epoch,
                             steps_per_epoch=steps_per_epoch,
                             steps_per_execution=500,
                             init_checkpoint=args.init_checkpoint)
    return run_config
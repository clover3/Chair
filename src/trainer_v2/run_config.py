import argparse
import sys


class RunConfigEx:
    def __init__(self,
        batch_size=16,
        train_step=0,
        eval_every_n_step=100,
        save_every_n_step=5000,
        learning_rate=2e-5,
        model_save_path="saved_model_ex",
        init_checkpoint="",
        checkpoint_type="bert",
        steps_per_epoch=-1,
        steps_per_execution=1
                 ):
        self.batch_size = batch_size
        self.train_step = train_step
        self.eval_every_n_step = eval_every_n_step
        self.save_every_n_step = save_every_n_step
        self.learning_rate = learning_rate
        self.model_save_path = model_save_path
        self.init_checkpoint = init_checkpoint
        self.checkpoint_type = checkpoint_type
        if steps_per_epoch == -1:
            self.steps_per_epoch = train_step
        else:
            self.steps_per_epoch = steps_per_epoch
        self.steps_per_execution = steps_per_execution


    def get_epochs(self):
        return self.train_step // self.steps_per_epoch

def get_run_config() -> RunConfigEx:
    parser = argparse.ArgumentParser(description='File should be stored in ')
    parser.add_argument("--model_save_path")
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--checkpoint_type")
    args = parser.parse_args(sys.argv[1:])
    run_config = RunConfigEx()
    nli_train_data_size = 392702
    step_per_epoch = int(nli_train_data_size / run_config.batch_size)

    if args.model_save_path:
        run_config.model_save_path = args.model_save_path

    run_config.checkpoint_type = args.checkpoint_type

    if args.init_checkpoint:
        run_config.init_checkpoint = args.init_checkpoint

    if run_config.checkpoint_type in ["none", "bert"]:
        num_epoch = 2
    elif run_config.checkpoint_type == "nli_saved_model":
        num_epoch = 1
    else:
        assert False

    run_config.train_step = num_epoch * step_per_epoch
    return run_config


class ExTrainConfig:
    num_deletion = 20
    g_val = 0.5
    save_train_payload = False
    drop_thres = 0.3
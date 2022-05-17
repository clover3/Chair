import json

from trainer_v2.chair_logging import c_log


class RunConfigEx:
    def __init__(self,
                 batch_size=16,
                 train_step=0,
                 train_epochs=0,
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

        if train_step:
            self.train_step = train_step
            if train_epochs > 0:
                raise ValueError("Only one of train_step or train_epochs should be specified")
        elif train_epochs:
            self.train_step = train_epochs * self.steps_per_epoch
        else:
            raise ValueError("One of train_step or train_epochs should be specified")

        self.steps_per_execution = steps_per_execution

    def get_epochs(self):
        return self.train_step // self.steps_per_epoch


class ExTrainConfig:
    num_deletion = 20
    g_val = 0.5
    save_train_payload = False
    drop_thres = 0.3


def get_run_config_nli_train(args):
    nli_train_data_size = 392702
    num_epochs = 4
    if args.config_path is not None:
        config_j = json.load(open(args.config_path, "r"))
    else:
        config_j = {}
    if 'batch_size' in config_j:
        batch_size = config_j['batch_size']
    else:
        batch_size = 16

    steps_per_epoch = int(nli_train_data_size / batch_size)
    run_config = RunConfigEx(model_save_path=args.output_dir,
                             train_step=num_epochs * steps_per_epoch,
                             steps_per_execution=1,
                             steps_per_epoch=steps_per_epoch,
                             init_checkpoint=args.init_checkpoint)

    for key, value in config_j.items():
        if key not in run_config.__dict__:
            c_log.warn("Key '{}' is not in the run config".format(key))
        run_config.__setattr__(key, value)
        c_log.info("Overwrite {} as {}".format(key, value))
    return run_config
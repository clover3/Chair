import json

from trainer_v2.chair_logging import c_log


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


class ExTrainConfig:
    num_deletion = 20
    g_val = 0.5
    save_train_payload = False
    drop_thres = 0.3


def get_run_config_nli_train(args):
    nli_train_data_size = 392702
    num_epochs = 4
    config_j = json.load(open(args.config_path, "r"))
    if 'batch_size' in config_j:
        batch_size = config_j['batch_size']
    else:
        batch_size = 16

    steps_per_epoch = int(nli_train_data_size / batch_size)
    run_config = RunConfigEx(model_save_path=args.output_dir,
                             train_step=num_epochs * steps_per_epoch,
                             steps_per_execution=500,
                             steps_per_epoch=steps_per_epoch,
                             init_checkpoint=args.init_checkpoint)

    for key, value in config_j.items():
        if key not in run_config.__dict__:
            c_log.warn("Key {} is not in run config".format(key))
        run_config.__setattr__(key, value)
        c_log.info("Overwrite {} as {}".format(key, value))

    run_config.steps_per_execution = 10

    return run_config
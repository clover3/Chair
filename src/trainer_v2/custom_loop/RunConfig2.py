import json
from abc import ABC

from trainer_v2.chair_logging import c_log


class SubConfig(ABC):
    def print_info(self):
        c_log.info("{}".format(self.__dict__))


class InputFileConfig(SubConfig):
    def __init__(self, train_files_path: str, eval_files_path: str):
        self.train_files_path = train_files_path
        self.eval_files_path = eval_files_path


class TrainConfig(SubConfig):
    def __init__(self,
                 train_step=0,
                 train_epochs=0,
                 learning_rate=2e-5,
                 steps_per_epoch=-1,
                 eval_every_n_step=100,
                 save_every_n_step=5000,
                 model_save_path="saved_model_ex",
                 init_checkpoint="",
                 checkpoint_type="bert",
                 ):
        self.learning_rate = learning_rate
        self.eval_every_n_step = eval_every_n_step
        self.save_every_n_step = save_every_n_step
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

    def get_epochs(self):
        return self.train_step // self.steps_per_epoch

    @classmethod
    def default(cls):
        return TrainConfig(0, 0, 2e-5, -1)

    def print_info(self):
        if self.init_checkpoint is None:
            c_log.warning("No checkpoint specified!")


class EvalConfig(SubConfig):
    def __init__(self,
                 eval_step=-1,
                 model_save_path="saved_model_ex",
                 ):
        self.model_save_path = model_save_path
        self.eval_step = eval_step

    def print_info(self):
        c_log.info("Model to evaluate: {}".format(self.model_save_path))
        if self.eval_step > 1:
            c_log.info("eval_steps: {}".format(self.eval_step))
        else:
            c_log.info("eval_steps: all")


class CommonRunConfig(SubConfig):
    def __init__(self,
                 batch_size=16,
                 steps_per_execution=1,
                 is_debug_run=False,
                 ):
        self.batch_size = batch_size
        self.steps_per_execution = steps_per_execution
        self.is_debug_run = is_debug_run

    def print_info(self):
        if self.is_debug_run:
            c_log.warning("DEBUGGING in use")


class TPUConfig(SubConfig):
    def __init__(self, use_tpu, tpu_name):
        self.use_tpu = use_tpu
        self.tpu_name = tpu_name


class RunConfig2:
    def __init__(self,
                 common_run_config: CommonRunConfig,
                 input_file_config: InputFileConfig,
                 train_config: TrainConfig=None,
                 tpu_config=None,
                 eval_config: EvalConfig=None
                 ):
        self.common_run_config = common_run_config
        self.train_config = train_config
        self.eval_config: EvalConfig = eval_config
        self.tpu_config = tpu_config
        self.input_file_config = input_file_config
        self.sub_configs = []

    def get_sub_configs(self):
        all_configs = [self.common_run_config,
                       self.train_config, self.eval_config,
                       self.tpu_config, self.input_file_config]
        return [config for config in all_configs if config is not None]

    def is_training(self) -> bool:
        return self.train_config is not None

    def get_epochs(self) -> int:
        return self.train_config.get_epochs()

    def print_info(self):
        for sub_config in self.get_sub_configs():
            sub_config.print_info()

        if self.tpu_config is not None:
            self.tpu_config.print_info()
            if self.common_run_config.steps_per_execution == 1 and not self.common_run_config.is_debug_run:
                c_log.warning("Using tpu with steps_per_execution == 1")

    def get(self, key):
        sub_config = self.get_matching_sub_config(key)
        if sub_config:
            raise KeyError(key)
        return sub_config.__getattribute__(key)

    def get_matching_sub_config(self, key):
        for sub_config in self.get_sub_configs():
            if key in sub_config.__dict__:
                return sub_config
        return None



def get_run_config2_nli_train(args):
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
    train_config = TrainConfig(
        model_save_path=args.output_dir,
        train_step=num_epochs * steps_per_epoch,
        steps_per_epoch=steps_per_epoch,
        init_checkpoint=args.init_checkpoint
    )
    common_run_config = CommonRunConfig(
        steps_per_execution=1,
    )
    input_file_config = get_input_file_config(args)

    tpu_config = get_tpu_config(args)

    run_config = RunConfig2(common_run_config=common_run_config,
                            input_file_config=input_file_config,
                            train_config=train_config,
                            tpu_config=tpu_config
                            )

    update_run_config(config_j, run_config)
    return run_config


def get_input_file_config(args):
    input_file_config = InputFileConfig(
        args.input_files,
        args.eval_input_files
    )
    return input_file_config


def update_run_config(config_j, run_config):
    for key, value in config_j.items():
        sub_config = run_config.get_matching_sub_config(key)
        if sub_config is None:
            c_log.warn("Key '{}' is not in the run config".format(key))
        else:
            sub_config.__setattr__(key, value)
            c_log.info("Overwrite {} as {}".format(key, value))


def get_run_config2_nli_eval(args):
    if args.config_path is not None:
        config_j = json.load(open(args.config_path, "r"))
    else:
        config_j = {}

    common_run_config = CommonRunConfig(
        steps_per_execution=1,
    )
    input_file_config = InputFileConfig(
        args.input_files,
        args.eval_input_files
    )
    eval_config = EvalConfig(
        model_save_path=args.output_dir,
    )
    tpu_config = get_tpu_config(args)

    run_config = RunConfig2(common_run_config=common_run_config,
                            input_file_config=input_file_config,
                            eval_config=eval_config,
                            tpu_config=tpu_config
                            )

    update_run_config(config_j, run_config)
    return run_config


def get_tpu_config(args):
    if args.use_tpu:
        tpu_config = TPUConfig(args.use_tpu, args.tpu_name)
    else:
        tpu_config = None
    return tpu_config

import copy
import json
import os
import subprocess
from abc import ABC

from google_wrap.gs_wrap import get_last_model_path

STATUS_WAIT = 0
STATUS_RUNNING = 1
STATUS_COMPLETED = 3
STATUS_CANCELLED = 4


class Task(ABC):
    def __init__(self):
        self.process_name = None # Python module
        self.status = None #
        self.resource = None # resources that this task occupy
        self.pid = None #
        self.std_output_path = None #
        self.wait_resource = None
        self.task_id = None
        self.env = None
        self.use_tpu = False
        self.argument_dict = {}
        self.task_name = None

    def update_argument(self, override_dict):
        for key, value in override_dict.items():
            self.argument_dict[key] = value


    @classmethod
    def from_dict(cls, json_object):
        task = Task()
        for key, value in json_object.items():
            task.__dict__[key] = value
        return task

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def get_param_str(self):
        return arg_dict_to_str(self.argument_dict)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class ChairTask(Task):
    def __init__(self):
        super(ChairTask, self).__init__()
        my_env = os.environ.copy()
        my_env["PYTHONPATH"] = "/home/lesterny/code/Chair/src"
        self.env= my_env


class LMLikeTask(ChairTask):
    def __init__(self, main_source, checkpoint_path, run_name):
        super(LMLikeTask, self).__init__()
        self.use_tpu = True
        self.main_source = main_source
        self.argument_dict = {
            "bert_config_file": "/home/lesterny/code/Chair/data/config/bert_config.json",
            "init_checkpoint": checkpoint_path,
            "output_dir": "gs://clovertpu/training/model/" + run_name,
            "save_checkpoints_steps": 5000,
            "is_bert_checkpoint": True,
            "max_seq_length": 512,
            "iterations_per_loop": 1000,
            "train_batch_size": 32,
            "learning_rate": "5e-5",
            "checkpoint_type": "v2",
            "num_train_steps": 36815,
            "do_train": None,
            "use_tpu": True,
            "repeat_data": True,
            "run_name":run_name
        }

"""
export PYTHONPATH=src
run_name=nli_um5_75K
lr=5e-5
seq_len=300
train_batch_size=32
python3 src/trainer/estimator_main_v2.py \
	--bert_config_file=data/config/bert_config.json \
	--dbert_config_file=data/config/dbert_config.json \
	--init_checkpoint=gs://clovertpu/training/model/unmasked_5/model.ckpt-75000 \
	--input_file=gs://clovertpu/training/nli300_cls/train \
	--output_dir=gs://clovertpu/training/model/$run_name \
	--save_checkpoints_steps=5000 \
	--is_bert_checkpoint=True \
	--max_seq_length=$seq_len \
	--iterations_per_loop=1000 \
	--train_batch_size=$train_batch_size \
	--learning_rate=$lr \
	--checkpoint_type=v2 \
	--num_train_steps=36815 \
	--do_train \
	--use_tpu=True \
	--repeat_data=True \
	--run_name=$run_name \
	--tpu_name=v2-tf2-2
"""

class ComputeFTask(ChairTask):
    def __init__(self):
        super(ComputeFTask, self).__init__()
        self.process_name = "/home/lesterny/tf20/bin/python"


def arg_dict_to_str(d):
    arg_str = ""
    for key, value in d.items():
        if value is None:
            arg_str += " --{}".format(key)
        else:
            arg_str += " --{}={}".format(key, str(value))

    return arg_str

class BERT2NLI_Train(ComputeFTask):
    def __init__(self, checkpoint_path, run_name):
        super(BERT2NLI_Train, self).__init__()
        self.parameter = NotImplemented
        self.status = None
        self.use_tpu = True
        self.argument_dict = {
            "bert_config_file":"/home/lesterny/code/Chair/data/config/bert_config.json",
            "init_checkpoint":checkpoint_path,
            "output_dir":"gs://clovertpu/training/model/" + run_name,
            "save_checkpoints_steps":5000,
            "is_bert_checkpoint":True,
            "max_seq_length":300,
            "iterations_per_loop":1000,
            "train_batch_size":32,
            "learning_rate":"5e-5",
            "checkpoint_type":"v2",
            "num_train_steps":36815,
            "do_train":None,
            "use_tpu":True,
            "repeat_data": True,
            "run_name":run_name
        }


def get_last_model_path_from_dir_name(model_dir):
    model_dir_path = "training/model/" + model_dir
    return get_last_model_path(model_dir_path)


def log_task(task):
    task_info_dir = "task_info"
    file_path = os.path.join(task_info_dir, task.run_name)
    with open(file_path, "w") as f:
        json.dump(task.to_dict(), f)


def execute_lm_like_task_auto(task_source, init_checkpoint_dir_name, run_name):
    init_checkpoint = init_checkpoint_dir_name(init_checkpoint_dir_name)
    task = LMLikeTask(task_source, init_checkpoint, run_name)
    log_task(task)
    print(task.to_json_string())
    arg_str = task_source + " " + task.get_param_str()
    p = subprocess.Popen([task.process_name, arg_str], env=task.env, shell=True)



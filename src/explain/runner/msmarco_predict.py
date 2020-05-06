import argparse
import os
import pickle
import sys
from typing import List

from data_generator.job_runner import WorkerInterface, JobRunner
from explain.msmarco import Hyperparam, ExTrainConfig
from explain.nli_ex_predictor import NLIExPredictor
from misc_lib import tprint
from tf_util.enum_features import load_record
from tlm.data_gen.feature_to_text import take
from trainer.np_modules import get_batches_ex

parser = argparse.ArgumentParser(description='')
parser.add_argument("--input_dir", help="Input dir.")
parser.add_argument("--model_path", help="Your model path.")
parser.add_argument("--save_dir", help="Your save dir.")
parser.add_argument("--modeling_option")


def tfrecord_to_old_stype(tfrecord_path, feature_names: List):
    all_insts = []
    for feature in load_record(tfrecord_path):
        inst = []
        for key in feature_names:
            v = take(feature[key])
            inst.append(list(v))
        all_insts.append(inst)
    return all_insts


class PredictWorker(WorkerInterface):
    def __init__(self, input_dir, out_dir):
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.tag = "relevant"

    def load_model(self, hparam, tf_setting, model_path, modeling_option):
        self.hparam = hparam
        self.predictor = NLIExPredictor(hparam, tf_setting, model_path, modeling_option, [self.tag])


    def work(self, job_id):
        input_path = os.path.join(self.input_dir, str(job_id))
        save_path = os.path.join(self.out_dir, str(job_id))
        all_feature_list = ["input_ids", "input_mask", "segment_ids", "data_id"]
        data = tfrecord_to_old_stype(input_path, all_feature_list)
        data_wo_id = list([(e[0], e[1], e[2]) for e in data])
        data_ids = list([e[3][0] for e in data])
        batches = get_batches_ex(data_wo_id, self.hparam.batch_size, 3)
        tprint("predict")
        ex_logits = self.predictor.predict_ex(self.tag, batches)
        tprint("done")
        assert len(ex_logits) == len(data)
        assert len(ex_logits) == len(data_ids)
        output_dict = {}
        for data_id, data_enry, scores in zip(data_ids, data, ex_logits):
            data_enry.append(scores)
            output_dict[int(data_id)] = list(data_enry)
        pickle.dump(output_dict, open(save_path, "wb"))


def run(args):
    tprint("msmarco run")
    hp = Hyperparam()
    nli_setting = ExTrainConfig()
    def worker_factory(out_dir):
        worker = PredictWorker(args.input_dir, out_dir)
        worker.load_model(hp, nli_setting, args.model_path, "co")
        return worker

    runner = JobRunner(args.save_dir, 605, "pc_tfrecord_ex", worker_factory)
    runner.auto_runner()



if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    run(args)
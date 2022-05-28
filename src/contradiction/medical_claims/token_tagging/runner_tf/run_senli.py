import argparse
import pickle
import sys

from cache import save_to_pickle
from data_generator.NLI import nli_info
from data_generator.shared_setting import BertNLI
from explain.nli_ex_predictor import NLIExPredictor
from models.transformer import hyperparams
from trainer.np_modules import get_batches_ex

parser = argparse.ArgumentParser(description='')
parser.add_argument("--tag", help="Your input file.")
parser.add_argument("--model_path", help="Your model path.")
parser.add_argument("--run_name", )
parser.add_argument("--data_path")
parser.add_argument("--modeling_option")


def d_to_triple(e):
    return e['input_ids'], e['input_mask'], e['segment_ids']


def predict_nli_ex(hparam, nli_setting, data_path, model_path, run_name, modeling_option):
    print("predict_nli_ex")
    print("Modeling option: ", modeling_option)
    dataset = pickle.load(open(data_path, "rb"))
    x_list = list(map(d_to_triple, dataset))

    data_id_list = list(map(lambda x: x['data_id'], dataset))
    batches = get_batches_ex(x_list, hparam.batch_size, 3)

    predictor = NLIExPredictor(hparam, nli_setting, model_path, modeling_option)
    pred_list = []
    for tag in nli_info.tags:
        ex_logits = predictor.predict_ex(tag, batches)
        ex_logits_joined = list(zip(data_id_list, ex_logits))
        pred_list.append(ex_logits_joined)
    save_to_pickle(pred_list, run_name)


def run(args):
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    predict_nli_ex(hp, nli_setting,
                   args.data_path,
                   args.common_model_dir_root,
                   args.run_name,
                   args.modeling_option,
                   )


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    run(args)

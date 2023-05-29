from dataclasses import dataclass
from typing import List, Iterable, Callable, Dict, Tuple, Set

from transformers.modeling_tf_outputs import TFBaseModelOutputWithPoolingAndCrossAttentions
import numpy as np
from cache import load_from_pickle, save_to_pickle
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_pairwise_dataset
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.per_project.tli.tli_probe.runner.dev_init_model import load_weights_from_hdf5
from trainer_v2.per_project.transparency.mmp.pairwise_modeling import get_transformer_pairwise_model
import tensorflow as tf
from trainer_v2.per_project.transparency.mmp.pairwise_modeling import ModelConfig, get_transformer_pairwise_model
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2, get_run_config_for_predict
from trainer_v2.per_project.transparency.mmp.probe.probe_network import ProbeAndAttention
from trainer_v2.train_util.arg_flags import flags_parser
import sys


def get_model_for_candidate_extraction(run_config, model_config):
    pairwise_model = get_transformer_pairwise_model(model_config, run_config)
    probe_and_attention = ProbeAndAttention(pairwise_model)
    model = probe_and_attention.model

    def name_mapping(name, _):
        return name.replace("tf_bert_for_sequence_classification/", "")

    load_weights_from_hdf5(
        model, run_config.predict_config.model_save_path, name_mapping, None)
    return probe_and_attention


def compare_outputs(cur_output, ref_output):
    for key in cur_output:
        v1 = cur_output[key]
        try:
            v2 = ref_output[key]
            if type(v1) == dict:
                for key2 in v1:
                    v1_ = v1[key2]
                    v2_ = v2[key2]
                    v_sum = np.sum(v1_)
                    err = np.sum(v1_ - v2_)
                    print(key, key2, v_sum, err)
            else:
                err = np.sum(v1 - v2)
                print(key, err)
        except KeyError as e:
            print(e)


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config_for_predict(args)
    run_config.print_info()
    model_config = ModelConfig()
    probe_and_attention = get_model_for_candidate_extraction(
        run_config,
        model_config,
    )

    def build_dataset(input_files, is_for_training):
        return get_pairwise_dataset(
            input_files, run_config, ModelConfig256_1(), is_for_training, add_dummy_y=False)

    dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
    dataset = dataset.take(1)

    model = probe_and_attention.model
    output = model.predict(dataset)
    save_to_pickle(output, "probe_and_attention")
    # ref_output = load_from_pickle("probe_inf_dev")
    # compare_outputs(output, ref_output)

    # TODO : Iterate each pairs,
    #   select query term if layer1 score is higher than layer0's score
    #   Take a document term with top attention score
    #   increase the count of (q_term, d_term)
    return NotImplemented


def get_q_term_loc_iter(input_ids):
    CLS_ID = 101
    SEP_ID = 102
    for i in range(len(input_ids)):
        if i == 0:
            assert input_ids[i] == CLS_ID
        else:
            if input_ids[i] == SEP_ID:
                return
            else:
                yield i




def extract_from_probe_attention(output_d):
    hidden_probe = output_d["probe_on_hidden"]
    attentions = output_d["attentions"]
    input_ids = output_d["input_ids"]
    valid_pair: List[Tuple[int, int]] = []
    threshold_d = {}
    for j in range(12):
        key_lower = f"layer_{j}"
        key_upper = f"layer_{j+1}"

        if key_lower in hidden_probe and key_upper:
            valid_pair.append((j, j+1))

    NI = NotImplemented

    @dataclass
    class ProbeInst:
        input_ids: np.array
        hidden_probe: Dict[int, np.array]
        attentions: List[np.array]

    item: ProbeInst = NI
    for layer_lower, layer_upper in valid_pair:
        threshold = threshold_d[layer_lower]
        itr = get_q_term_loc_iter(item.input_ids)
        hp_lower = item.hidden_probe[layer_lower]
        hp_upper = item.hidden_probe[layer_upper]
        for query_term_loc in itr:
            probe_upper: float = hp_upper[query_term_loc][0]
            probe_lower: float = hp_lower[query_term_loc][0]
            gap = probe_upper - probe_lower

            if gap > threshold:
                q_term = item.input_ids[query_term_loc]
                # [N_HEAD, M]
                from_attn = item.attentions[layer_lower][:, query_term_loc, :]
                to_attn = item.attentions[layer_lower][:, : , query_term_loc]










if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)



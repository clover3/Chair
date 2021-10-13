import argparse
import copy
import pickle
import sys
from typing import List
from typing import NamedTuple

import numpy as np
import tensorflow as tf

from cache import save_to_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.pairing.predict_middle_out import predict_middle_in
from explain.pairing.probe_model_middle_in import ClsProbeMiddleIn
from explain.pairing.probe_train_common import HPCommon, NLIPairingTrainConfig
from explain.pairing.runner.run_train_cls_probe import ClsProbeConfig
from explain.setups import init_fn_generic
from trainer.np_modules import get_batches_ex
from trainer.tf_train_module import init_session


class ReplaceOp(NamedTuple):
    vector_idx: int
    source_seq_loc: int
    target_seq_loc: int


def load_mapping_info(path):
    # Each line is an instance
    def parse_line(line):
        tokens = line.split("\t")
        base_vector_idx = int(tokens[0])
        replace_sequence: List[ReplaceOp] = []
        for token in tokens[1:]:
            vector_idx, source_loc, target_loc = token.split(",")
            ro = ReplaceOp(int(vector_idx), int(source_loc), int(target_loc))
            replace_sequence.append(ro)
        return base_vector_idx, replace_sequence
    return list(map(parse_line, open(path, "r")))


def prepare_data(vector_path, mapping_path):
    source = pickle.load(open(vector_path, "rb"))
    tokenizer = get_tokenizer()
    mapping_info = load_mapping_info(mapping_path)

    key_hidden_vector = "other_value_0"
    key_attn_mask = "other_value_1"
    output = []
    def convert_id(id_elem):
        return tokenizer.convert_ids_to_tokens([id_elem])[0]

    for inst_idx, instance_info in enumerate(mapping_info):
        base_vector_idx, replace_sequence = instance_info
        base_vector = copy.deepcopy(source[key_hidden_vector][base_vector_idx])
        base_attn_mask = copy.deepcopy(source[key_attn_mask][base_vector_idx])
        input_ids = copy.deepcopy(source["input_ids"][base_vector_idx])
        input_mask = copy.deepcopy(source["input_mask"][base_vector_idx])
        segment_ids = copy.deepcopy(source["segment_ids"][base_vector_idx])

        print("instance {} base_idx={}".format(inst_idx, base_vector_idx))
        for replace_op in replace_sequence:
            src_idx = replace_op.source_seq_loc
            trg_idx = replace_op.target_seq_loc
            vector = copy.deepcopy(source[key_hidden_vector][replace_op.vector_idx])
            print(np.shape(vector))
            source_input_id = source["input_ids"][replace_op.vector_idx]
            base_vector[trg_idx] = vector[src_idx]
            print("replace {}-th tokens ({}) with {}-th token of {}-th vector ({})".format(
                trg_idx,
                convert_id(input_ids[trg_idx]),
                src_idx,
                replace_op.vector_idx,
                convert_id(source_input_id[src_idx])
            ))
            input_ids[trg_idx] = source_input_id[src_idx]

        output.append((input_ids, input_mask, segment_ids, base_vector, base_attn_mask))
    return output


def do_predict(bert_hp, train_config, dev_batches,
               lms_config, init_fn, middle_layer
               ):
    num_gpu = train_config.num_gpu
    model = ClsProbeMiddleIn(bert_hp, lms_config, num_gpu,
                             middle_layer, False, True)
    other_tensor_list = []
    sess = init_session()
    sess.run(tf.global_variables_initializer())
    init_fn(sess)
    output_d = predict_middle_in(sess, dev_batches,
                                 model.logits,
                                 model.loss_tensor,
                                 model.per_layer_logit_tensor,
                                 other_tensor_list,
                                 model.embedding_feed_dict)
    return output_d


def main(start_model_path,
         modeling_option,
         vector_path,
         mapping_path,
         save_name, middle_layer, num_gpu=1):
    middle_layer = int(middle_layer)
    num_gpu = int(num_gpu)
    hp = HPCommon()
    train_config = NLIPairingTrainConfig()
    train_config.num_gpu = num_gpu

    def init_fn(sess):
        return init_fn_generic(sess, "as_is", start_model_path)
    probe_config = ClsProbeConfig()
    probe_config.per_layer_component = modeling_option

    flat_data = prepare_data(vector_path, mapping_path)
    batches = get_batches_ex(flat_data, HPCommon.batch_size, 5)

    x0, x1, x2, encoded_embedding_in, attention_mask = batches[0]
    for j in range(5):
        print(np.sum(encoded_embedding_in[j][22]))
    output_d = do_predict(hp, train_config, batches,
                          probe_config, init_fn, middle_layer)
    save_to_pickle(output_d, save_name)


ex_arg_parser = argparse.ArgumentParser(description='File should be stored in ')
ex_arg_parser.add_argument("--start_model_path", help="Your input file.")
ex_arg_parser.add_argument("--modeling_option")
ex_arg_parser.add_argument("--num_gpu", default=1)
ex_arg_parser.add_argument("--vector_path")
ex_arg_parser.add_argument("--mapping_path")
ex_arg_parser.add_argument("--middle_layer")
ex_arg_parser.add_argument("--save_name")


if __name__ == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    main(args.start_model_path,
         args.modeling_option,
         args.vector_path,
         args.mapping_path,
         args.save_name,
         args.middle_layer,
         args.num_gpu)

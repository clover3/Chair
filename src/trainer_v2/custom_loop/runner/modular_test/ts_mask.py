import sys
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.neural_network_def.var_local_decisions import SingleVarLD, transform_inputs_for_ts
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser


class ModelConfig(ModelConfigType):
    max_seq_length = 300
    num_classes = 3
    num_local_classes = 3


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.print_info()

    model_config = ModelConfig()

    def build_dataset(input_files, is_for_training):
        return get_classification_dataset(input_files, run_config, model_config, is_for_training)

    train_dataset = build_dataset(run_config.dataset_config.train_files_path, True)
    train_itr = iter(train_dataset)

    for x, y in train_itr:
        l_input_ids, l_token_type_ids = x
        print(y.shape)
        print(l_input_ids.shape)
        print(l_token_type_ids.shape)
        outputs = transform_inputs_for_ts(l_input_ids, l_token_type_ids)
        n_input_ids, n_segment_ids = outputs
        print(n_input_ids.shape)
        print(n_segment_ids.shape)
        for i in range(16):
            print("before", l_input_ids[i][:50].numpy().tolist())
            print("after", n_input_ids[i][:50].numpy().tolist())
            print("before", l_token_type_ids[i][:50].numpy().tolist())
            print("after", n_segment_ids[i][:50].numpy().tolist())
            print("-----")
        break



if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)



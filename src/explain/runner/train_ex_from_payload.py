import sys

import data_generator.NLI.nli_info
from explain.runner.nli_ex_param import ex_arg_parser
from explain.runner.train_ex import get_params, train_nli_ex_from_payload


def train_from(start_model_path, start_type, save_dir, modeling_option, tags, info_fn_name, num_gpu=1):
    data, data_loader, hp, informative_fn, init_fn, train_config\
        = get_params(start_model_path, start_type, info_fn_name, num_gpu)
    train_config.max_steps = 73630

    train_nli_ex_from_payload(hp, train_config, save_dir,
                 data, data_loader, tags, modeling_option,
                 init_fn)


if __name__  == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    train_from(args.start_model_path,
               args.start_type,
               args.save_dir,
               args.modeling_option,
               data_generator.NLI.nli_info.tags,
               args.info_fn,
               int(args.num_gpu))

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


flags_parser = argparse.ArgumentParser(description='')
flags_parser.add_argument("--tpu_name", help="The Cloud TPU to use for training. This should be either the name "
                                        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
                                        "url.")
flags_parser.add_argument("--use_tpu", type=str2bool, nargs='?',
                          const=True, default=False,
                          help="Whether to use TPU or GPU/CPU.")
flags_parser.add_argument("--input_files", )
flags_parser.add_argument("--eval_input_files", )
flags_parser.add_argument("--init_checkpoint", )
flags_parser.add_argument("--config_path", )
flags_parser.add_argument("--output_dir", )
flags_parser.add_argument("--action", default="train")

flags_parser.add_argument("--job_id", type=int, default=-1)
flags_parser.add_argument("--run_name", default=None)

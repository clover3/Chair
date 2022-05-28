
import sys

from data_generator.NLI.nli import get_modified_data_loader2
from data_generator.shared_setting import BertNLI
from explain.runner.predict_params import parser
from explain.runner.run_predict import predict_nli_ex
from models.transformer import hyperparams


def run(args):
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    data_loader = get_modified_data_loader2(hp, nli_setting)

    predict_nli_ex(hp, nli_setting, data_loader,
                   args.tag,
                   args.data_id,
                   args.common_model_dir_root,
                   args.run_name,
                   args.modeling_option,
                   [args.tag],
                   )


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    run(args)
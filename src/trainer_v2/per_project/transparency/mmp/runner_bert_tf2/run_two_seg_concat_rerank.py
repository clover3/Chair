import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_dev_helper import mmp_rerank_dev_set
from trainer_v2.per_project.transparency.mmp.eval_helper.pep_rerank import get_pep_scorer_from_pointwise
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    build_scorer_fn = get_pep_scorer_from_pointwise
    mmp_rerank_dev_set(args, build_scorer_fn)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

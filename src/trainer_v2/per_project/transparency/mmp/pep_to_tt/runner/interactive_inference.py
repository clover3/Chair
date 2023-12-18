import sys

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.inf_helper import PEP_TT_Inference
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


def main():
    c_log.info(__file__)
    model_path = sys.argv[1]
    strategy = get_strategy()
    model_config = PEP_TT_ModelConfig()
    with strategy.scope():
        inf_helper = PEP_TT_Inference(model_config, model_path)

        while True:
            query = input("Enter query term: ")
            doc = input("Enter document term: ")
            ret = inf_helper.score_fn([(query, doc)])[0]
            print(ret)


if __name__ == "__main__":
    main()

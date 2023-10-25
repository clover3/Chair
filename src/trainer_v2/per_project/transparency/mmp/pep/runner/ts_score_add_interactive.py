



import sys

from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import tokenize_w_mask_preserving
from trainer_v2.per_project.transparency.mmp.pep.demo_util import get_pep_local_decision
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    c_log.info(__file__)
    model_path = sys.argv[1]
    strategy = get_strategy()
    tokenizer = get_tokenizer()

    def tokenize(text):
        return tokenize_w_mask_preserving(tokenizer, text)

    with strategy.scope():
        score_fn = get_pep_local_decision(model_path)
        while True:
            query = input("Enter query segment: ")
            doc = input("Enter document part: ")
            t = tokenize(query), tokenize(doc)
            ret = score_fn(t)
            local_d, global_d = ret

            target_l = local_d[0][0]
            print(ret)
            print(float(target_l))


if __name__ == "__main__":
    main()

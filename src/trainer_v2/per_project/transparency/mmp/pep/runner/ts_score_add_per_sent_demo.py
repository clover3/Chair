




import sys

from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import get_second
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
            query = input("Enter full query: ")
            doc = input("Enter document part: ")
            q_tokens = query.split()
            d_tokens = doc.split()
            for q_token in q_tokens:
                output = []
                for d_token in d_tokens:
                    query_rep = "[MASK] {} [MASK]".format(q_token)

                    head = " ".join(["[MASK]"] * 4)
                    tail = " ".join(["[MASK]"] * 12)
                    doc_rep = head + " " + d_token + " " + tail
                    t = tokenize(query_rep), tokenize(doc_rep)
                    score: float = score_fn(t)
                    output.append((d_token, score))

                output.sort(key=get_second, reverse=True)
                print(q_token)
                print(output[:3])



if __name__ == "__main__":
    main()

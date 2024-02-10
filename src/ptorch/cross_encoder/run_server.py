from ptorch.cross_encoder.get_ce_msmarco_mini_lm import get_ce_msmarco_mini_lm_score_fn
from rpc.bert_like_server import RPCServerWrap


def main():
    port = 28122
    predict_fn = get_ce_msmarco_mini_lm_score_fn()
    server = RPCServerWrap(predict_fn)
    server.start(port)


if __name__ == "__main__":
    main()
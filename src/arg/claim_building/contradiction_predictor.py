import sys

from data_generator.shared_setting import BertNLI
from explain.nli_ex_predictor import NLIExPredictor
from models.transformer import hyperparams
from rpc.bert_like_server import BertLikeServer

PORT_CONFLICT_EX = 8122


def run_server(model_path):
    hparam = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    modeling_option = "co"
    predictor = NLIExPredictor(hparam, nli_setting, model_path, modeling_option)

    explain_tag = 'conflict'

    # payload is list of (input_ids, input_mask, segment_ids)
    def predict(payload):
        sout, ex_logits = predictor.predict_both_from_insts(explain_tag, payload)
        sout = sout.tolist()
        ex_logits = ex_logits.tolist()
        return list(zip(sout, ex_logits))

    server = BertLikeServer(predict)
    print("server started")
    server.start(PORT_CONFLICT_EX)


if __name__ == "__main__":
    run_server(sys.argv[1])
from cpath import get_bert_full_path
from tf_v2_support import disable_eager_execution
from tlm.benchmark.nli_v2 import run_nli_w_path


def fn():
    disable_eager_execution()
    model_path = get_bert_full_path()
    run_nli_w_path("bert_nli", "0", model_path)


if __name__ == "__main__":
    fn()
from rpc.disk_dump import dump_dict
from tlm.robust_tokens import load_robust_token
from data_generator.data_parser import trec
import os, path
from data_generator import tokenizer_wo_tf as tokenization

def dump_robust():
    c = trec.load_robust_ingham()
    dump_dict(c, "robust")

def dump_tokens():
    token_d = load_robust_token()
    dump_dict(token_d, "robust_token")


def dump_robust_cap_tokens():
    vocab_file = os.path.join(path.data_path, "bert_voca.txt")
    cap_tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=False)

    c = trec.load_robust_ingham()
    d = {}
    for key in c:
        doc = c[key]
        d[key] = cap_tokenizer.basic_tokenizer.tokenize(doc)
    dump_dict(d, "robust_token_cap")



#dump_robust()
dump_robust_cap_tokens()



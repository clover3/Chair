from rpc.disk_dump import dump_dict
from tlm.robust_tokens import load_robust_token
from data_generator.data_parser import trec

def dump_robust():
    c = trec.load_robust_ingham()
    dump_dict(c, "robust")

def dump_tokens():
    token_d = load_robust_token()
    dump_dict(token_d, "robust_token")


dump_robust()
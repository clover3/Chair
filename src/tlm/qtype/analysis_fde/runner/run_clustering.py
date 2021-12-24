from misc_lib import tprint
from tlm.qtype.analysis_fde.runner.build_q_emb_from_samples import load_q_emb_qtype_2X_v_train_200000
from tlm.qtype.analysis_qde.analysis_a import cluster_avg_embeddings


def main():
    tprint("Loading pickle...")
    emb_d = load_q_emb_qtype_2X_v_train_200000()
    tuple_list = [(k, v) for k, v in emb_d.items()]
    cluster_avg_embeddings(tuple_list)


if __name__ == "__main__":
    main()
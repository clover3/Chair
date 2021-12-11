import os

from cpath import output_path
from tf_util.enum_features import load_record
from tlm.qtype.analysis_qde.qde_sentence_perturb import iter_de_tokens


def main():
    tfrecord_path = os.path.join(output_path, "MMD_train_qe_de_distill_10doc", "0")
    records = load_record(tfrecord_path)
    iter_de_tokens(records)


if __name__ == "__main__":
    main()
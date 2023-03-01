from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from epath import job_man_dir
from misc_lib import path_join, pause_hook
from tf_util.enum_features import load_record_v2
from tf_util.tfrecord_convertor import take


def main():
    tfrecord_path = path_join(job_man_dir,  "MMD_passage_based_train", "0")

    itr = load_record_v2(tfrecord_path)
    tokenizer = get_tokenizer()

    def parse_from_ids(input_ids):
        text = ids_to_text(tokenizer, input_ids)
        q, d, dummy = text.split("[SEP]")
        q = q.replace("[CLS] ", "")
        return q, d

    for features in pause_hook(itr, 20):
        input_ids1 = take(features["input_ids1"])
        input_ids2 = take(features["input_ids2"])

        q1, d1 = parse_from_ids(input_ids1)
        q2, d2 = parse_from_ids(input_ids2)

        assert q1 == q2
        print("Query: " + q1)
        print("Pos: " +  d1)
        print("Neg: " + d2)
        print()


    return NotImplemented


if __name__ == "__main__":
    main()
import os
import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from epath import job_man_dir
from estimator_helper.output_reader import load_combine_info_jsons


def main():
    info_path = os.path.join(job_man_dir, "MMD_pred_info", "1.info")
    info = load_combine_info_jsons(info_path)
    tokenizer = get_tokenizer()
    cnt = 0
    fn = os.path.join(job_man_dir, "MMD_pred", "1")

    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        keys = feature.keys()

        print("---- record -----")
        v = feature["input_ids"].int64_list.value
        data_id = feature["data_id"].int64_list.value[0]
        info_entry = info[str(data_id)]
        passage_idx = info_entry['passage_idx']
        tokens = tokenizer.convert_ids_to_tokens(v)
        text = " ".join(tokens)
        sep_idx = text.find("[SEP]")
        print(passage_idx)
        print(text[:sep_idx])
        print(text[sep_idx:])

        cnt += 1
        if cnt >= 10:
            break


if __name__ == "__main__":
    main()
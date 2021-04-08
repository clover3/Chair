import json
from typing import Iterable

from nltk.tokenize import sent_tokenize

from arg.counter_arg.point_counter.tf_encoder import get_encode_fn_w_data_id
from arg.counter_arg_retrieval.f5.load_f5_clue_docs import load_f5_docs_texts
from cpath import at_output_dir
from misc_lib import DataIDManager
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.classification_common import TextInstance


def enum_f5_data() -> Iterable[str]:
    texts = load_f5_docs_texts()
    for text in texts:
        yield text
        for sent in sent_tokenize(text):
            yield sent


def main():
    data_id_manager = DataIDManager()
    data = []
    for text in enum_f5_data():
        info = {
            'text': text,
        }
        data_id = data_id_manager.assign(info)
        label = 0
        data.append(TextInstance(text, label, data_id))

    encode_fn = get_encode_fn_w_data_id(512, False)
    save_path = at_output_dir("clue_counter_arg", "clue_f5.tfrecord")
    write_records_w_encode_fn(save_path, encode_fn, data)

    info_save_path = at_output_dir("clue_counter_arg", "clue_f5.tfrecord.info")
    json.dump(data_id_manager.id_to_info, open(info_save_path, "w"))


if __name__ == "__main__":
    main()
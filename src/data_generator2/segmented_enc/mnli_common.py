from dataset_specific.mnli.mnli_reader import MNLIReader
from tf_util.record_writer_wrap import write_records_w_encode_fn


def mnli_encode_common(encode_fn, split, output_path):
    data_size = 400 * 1000 if split == "train" else 10000
    reader = MNLIReader()
    write_records_w_encode_fn(output_path, encode_fn, reader.load_split(split), data_size)

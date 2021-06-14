import collections
import os
from typing import List, Callable, Tuple, Iterator
from typing import NewType, Any

from misc_lib import get_dir_files, exist_or_mkdir
from tf_util.enum_features import load_record_v2
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take

TFRecordFeature = NewType("TFRecordFeature", Any)
InputSegmentMaskIdsTuple = NewType("InputSegmentMaskIdsTuple", Tuple)


def convert(input_rep: InputSegmentMaskIdsTuple, source_idx) -> collections.OrderedDict:
    input_ids, input_mask, segment_ids = input_rep
    new_features = collections.OrderedDict()
    new_features["input_ids"] = create_int_feature(input_ids)
    new_features["input_mask"] = create_int_feature(input_mask)
    new_features["segment_ids"] = create_int_feature(segment_ids)
    new_features["label_ids"] = create_int_feature([source_idx])
    return new_features


def write_classification_tfrecord(source_itr_list: List[Iterator[InputSegmentMaskIdsTuple]],
                                  output_dir,
                                  record_per_file=1000
                                  ):
    out_file_idx = 0
    exist_or_mkdir(output_dir)

    def get_next_writer():
        nonlocal out_file_idx
        output_path = os.path.join(output_dir, str(out_file_idx))
        writer = RecordWriterWrap(output_path)
        out_file_idx += 1
        return writer

    writer = get_next_writer()
    try:
        while True:
            for source_idx, itr in enumerate(source_itr_list):
                item: InputSegmentMaskIdsTuple = itr.__next__()  # This would raise StopIteration
                new_features = convert(item, source_idx)
                writer.write_feature(new_features)
                if writer.total_written >= record_per_file:
                    writer.close()
                    writer = get_next_writer()
    except StopIteration:
        pass


def single_input_tfrecord_reader(record: TFRecordFeature) -> Iterator[InputSegmentMaskIdsTuple]:
    input_ids = take(record["input_ids"])
    input_mask = take(record["input_mask"])
    segment_ids = take(record["segment_ids"])
    yield input_ids, input_mask, segment_ids


def pair_input_tfrecord_reader(record: TFRecordFeature) -> Iterator[InputSegmentMaskIdsTuple]:
    input_ids = take(record["input_ids1"])
    input_mask = take(record["input_mask1"])
    segment_ids = take(record["segment_ids1"])
    yield input_ids, input_mask, segment_ids

    input_ids = take(record["input_ids2"])
    input_mask = take(record["input_mask2"])
    segment_ids = take(record["segment_ids2"])
    yield input_ids, input_mask, segment_ids


def mes_input_tfrecord_reader(record: TFRecordFeature) -> Iterator[InputSegmentMaskIdsTuple]:
    input_ids = take(record["input_ids"])
    input_mask = take(record["input_mask"])
    segment_ids = take(record["segment_ids"])
    window_size = 512
    n_window = 4
    assert len(input_ids) == window_size * n_window
    for i in range(n_window):
        st = i * window_size
        ed = st + window_size
        yield input_ids[st:ed], input_mask[st:ed], segment_ids[st:ed]


def iter_dir_record(dir_path,
                    tfrecord_read_fn: Callable[[TFRecordFeature], Iterator[TFRecordFeature]])\
        -> Iterator[TFRecordFeature]:
    itr_list = []
    for file_path in get_dir_files(dir_path):
        tfrecord_itr: Iterator[TFRecordFeature] = load_record_v2(file_path)
        itr_list.append(tfrecord_itr)

    while itr_list:
        for itr in itr_list:
            try:
                tfrecord: TFRecordFeature = itr.__next__()
                yield from tfrecord_read_fn(tfrecord)
            except StopIteration:
                itr_list.remove(itr)


def make_classification_from_pair_input_triplet(train_dir_path, dev_dir_path, test_dir_path, output_dir):
    source_list = [
        (train_dir_path, pair_input_tfrecord_reader),
        (dev_dir_path, single_input_tfrecord_reader),
        (test_dir_path, single_input_tfrecord_reader),
    ]

    source_itr_list = []
    for dir_path, input_tfrecord_reader in source_list:
        itr: Iterator[TFRecordFeature] = iter_dir_record(dir_path, input_tfrecord_reader)
        source_itr_list.append(itr)

    write_classification_tfrecord(source_itr_list, output_dir)


def make_classification_from_single_input_triplet(train_dir_path, dev_dir_path, test_dir_path, output_dir):
    source_list = [
        (train_dir_path, single_input_tfrecord_reader),
        (dev_dir_path, single_input_tfrecord_reader),
        (test_dir_path, single_input_tfrecord_reader),
    ]

    source_itr_list = []
    for dir_path, input_tfrecord_reader in source_list:
        itr: Iterator[TFRecordFeature] = iter_dir_record(dir_path, input_tfrecord_reader)
        source_itr_list.append(itr)

    write_classification_tfrecord(source_itr_list, output_dir)



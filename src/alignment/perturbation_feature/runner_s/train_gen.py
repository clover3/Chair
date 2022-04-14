from alignment.nli_align_path_helper import get_tfrecord_path
from alignment.perturbation_feature.data_gen import make_tf_feature
from alignment.perturbation_feature.segments_to_features import build_x_y
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape

from bert_api.task_clients.nli_interface.nli_interface import get_nli_cache_client

from tf_util.record_writer_wrap import RecordWriterWrap


def main():
    nli_client = get_nli_cache_client("localhost")
    dataset_name = "train"
    scorer_name = "lexical_v1"
    save_path = get_tfrecord_path(f"{dataset_name}_{scorer_name}")
    xy_iter = build_x_y(dataset_name, nli_client, scorer_name)
    shape = get_pert_train_data_shape()
    writer = RecordWriterWrap(save_path)
    for x, y in xy_iter:
        feature = make_tf_feature(x, y, shape)
        writer.write_feature(feature)


if __name__ == "__main__":
    main()
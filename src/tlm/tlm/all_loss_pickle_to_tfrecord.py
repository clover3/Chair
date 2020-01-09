import collections
import logging
import os
import pickle
import sys

from tf_util.record_writer_wrap import RecordWriterWrap
from tf_util.tf_logging import tf_logging
from tlm.data_gen.bert_data_gen import create_int_feature, create_float_feature
from tlm.estimator_prediction_viewer import flatten_batches

working_dir = os.environ["TF_WORKING_DIR"]


def load(job_id):
    file_path = os.path.join(working_dir, "manual_all_loss", "{}.pickle".format(job_id))
    if os.path.exists(file_path):
        return pickle.load(open(file_path, "rb"))
    else:
        print(file_path)
        return None


def work(job_id):
    outfile = os.path.join(working_dir, "BLC_data", "{}".format(job_id))
    if os.path.exists(outfile):
        return "Skip"
    tf_logging.debug("Loading data")
    data = load(job_id)
    tf_logging.debug("Done")
    if data is None:
        return "No Input"

    writer = RecordWriterWrap(outfile)

    batch_size, seq_length = data[0]['input_ids'].shape
    keys = list(data[0].keys())

    vectors = flatten_batches(data)
    basic_keys = "input_ids", "input_mask", "segment_ids"
    any_key = keys[0]
    data_len = len(vectors[any_key])
    num_predictions = len(vectors["grouped_positions"][0][0])

    for i in range(data_len):
        mask_valid = [0] * seq_length
        loss1_arr = [0] * seq_length
        loss2_arr = [0] * seq_length
        positions = vectors["grouped_positions"][i]
        num_trials = len(positions)
        for t_i in range(num_trials):
            for p_i in range(num_predictions):
                loc = vectors["grouped_positions"][i][t_i][p_i]
                loss1 = vectors["grouped_loss1"][i][t_i][p_i]
                loss2 = vectors["grouped_loss2"][i][t_i][p_i]

                loss1_arr[loc] = loss1
                loss2_arr[loc] = loss2
                assert mask_valid[loc] == 0
                mask_valid[loc] = 1

        features = collections.OrderedDict()
        for key in basic_keys:
            features[key] = create_int_feature(vectors[key][i])

        features["loss_valid"] = create_int_feature(mask_valid)
        features["loss1"] = create_float_feature(loss1_arr)
        features["loss2"] = create_float_feature(loss2_arr)
        features["next_sentence_labels"] = create_int_feature([0])
        writer.write_feature(features)
        #if i < 20:
        #    log_print_feature(features)
    writer.close()
    return "Done"


if __name__ == "__main__":
    tf_logging.setLevel(logging.INFO)
    st = int(sys.argv[1])
    ed = int(sys.argv[2])
    for i in range(st, ed):
        ret = work(i)
        print("Job {} {}".format(i, ret))

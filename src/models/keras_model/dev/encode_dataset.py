import os

# Load the required submodules
import tensorflow as tf
from official import nlp
from official.nlp import bert

from cpath import data_path


def main():
    processor = nlp.data.classifier_data_lib.TfdsProcessor(
        tfds_params="dataset=glue/mrpc,text_key=sentence1,text_b_key=sentence2",
        process_text_fn=bert.tokenization.convert_to_unicode)
    train_data_output_path="./mrpc_train.tf_record"
    eval_data_output_path="./mrpc_eval.tf_record"

    max_seq_length = 128
    batch_size = 32
    eval_batch_size = 32
    bert_voca_path = os.path.join(data_path, "bert_voca.txt")

    tokenizer = bert.tokenization.FullTokenizer(
        vocab_file=bert_voca_path,
        do_lower_case=True)
    # Generate and save training data into a tf record file
    input_meta_data = (
        nlp.data.classifier_data_lib.generate_tf_record_from_data_file(
            processor=processor,
            data_dir=None,  # It is `None` because data is from tfds, not local dir.
            tokenizer=tokenizer,
            train_data_output_path=train_data_output_path,
            eval_data_output_path=eval_data_output_path,
            max_seq_length=max_seq_length))
    training_dataset = bert.run_classifier.get_dataset(
        train_data_output_path,
        max_seq_length,
        batch_size,
        is_training=True)()

    evaluation_dataset = bert.run_classifier.get_dataset(
        eval_data_output_path,
        max_seq_length,
        eval_batch_size,
        is_training=False)()
    training_dataset.element_spec


def create_classifier_dataset(file_path, seq_length, batch_size, is_training):
    """Creates input dataset from (tf)records files for train/eval."""
    dataset = tf.data.TFRecordDataset(file_path)
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()

    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(record, name_to_features)

    def _select_data_from_record(record):
        x = {
            'input_word_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'input_type_ids': record['segment_ids']
        }
        y = record['label_ids']
        return (x, y)

    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        _select_data_from_record,
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    main()
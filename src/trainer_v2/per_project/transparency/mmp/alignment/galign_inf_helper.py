import tensorflow as tf


def build_dataset_q_term_d_term(q_term: int, d_term_id_st: int, d_term_id_ed: int):
    # Create a range of integers from st to ed
    data_range = tf.range(d_term_id_st, d_term_id_ed)

    # Create a tf.data.Dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices({
        'd_term': data_range
    })

    def add_q_term_make_array(x):
        x = {'d_term': [x['d_term']],
             'q_term': tf.constant([q_term], dtype=tf.int32),
             'raw_label': tf.zeros([1], dtype=tf.float32),
             'label': tf.zeros([1], dtype=tf.int32),
             'is_valid': tf.zeros([1], dtype=tf.int32),
             }
        return x

    # Add the q_term feature to each record in the dataset
    dataset = dataset.map(add_q_term_make_array)
    return dataset
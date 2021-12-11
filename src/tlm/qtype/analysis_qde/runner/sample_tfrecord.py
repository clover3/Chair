import sys

from tlm.qtype.analysis_fixed_qtype.sample_tfrecord import extract_selector, random_true


def main():
    def select_fn(record):
        if record['label_ids'][0] > 5:
            return random_true(0.02)
        else:
            return random_true(0.005)

    int_features = [
                    'd_e_input_ids', 'd_e_segment_ids',
                    'q_e_input_ids', 'q_e_segment_ids',
                    'data_id']
    float_features = ['label_ids']

    extract_selector(sys.argv[1], sys.argv[2], int_features, float_features, select_fn)


if __name__ == "__main__":
    main()
import csv

from data_generator.tokenizer_wo_tf import EncoderUnitPlain


class LightDataLoader:
    def __init__(self, max_sequence, voca_path):
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.encoder_unit = EncoderUnitPlain(max_sequence, voca_path)
        self.max_seq = max_sequence

    def example_generator(self, file_path):
        f = open(file_path, "r", encoding="utf-8", errors="ignore")
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            s1 = row[0]
            s2 = row[1]
            if len(row) > 2:
                label = int(row[2])
            else:
                label = 0
            input_ids, input_mask, segment_ids = self.encoder_unit.encode_pair(s1, s2)
            yield input_ids, input_mask, segment_ids, label

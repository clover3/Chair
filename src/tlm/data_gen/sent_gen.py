from list_lib import flatten
from tlm.data_gen.base import truncate_seq_pair, format_tokens_pair_n_segid
from tlm.data_gen.lm_datagen import UnmaskedGen, SegmentInstance


class SentGen(UnmaskedGen):
    def __init__(self):
        super(SentGen, self).__init__()

    def create_instances_from_documents(self, documents):
        max_num_tokens = self.max_seq_length - 3
        target_seq_length = max_num_tokens

        all_segment = flatten(documents)
        instances = []
        for seg in all_segment:
            tokens_a = seg
            tokens_b = []
            truncate_seq_pair(tokens_a, tokens_b, target_seq_length, self.rng)

            tokens, segment_ids = format_tokens_pair_n_segid(tokens_a, tokens_b)
            instance = SegmentInstance(
                tokens=tokens,
                segment_ids=segment_ids)
            instances.append(instance)

        return instances

    def write_instance_to_example_file(self, insts, output_file):
        return self.write_instance_to_example_files(insts, [output_file])
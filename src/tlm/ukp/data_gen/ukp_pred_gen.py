from data_generator.common import get_tokenizer
from misc_lib import flatten
from tlm.data_gen.base import UnmaskedGen, truncate_seq_pair, format_tokens_pair_n_segid, SegmentInstance


class UkpSentGen(UnmaskedGen):
    def __init__(self):
        super(UnmaskedGen, self).__init__()

    def create_instances_from_documents(self, topic, documents):
        tokenizer = get_tokenizer()
        topic_tokens = tokenizer.tokenize(topic)

        max_num_tokens = self.max_seq_length - 3
        target_seq_length = max_num_tokens

        all_segment = flatten(documents)
        instances = []
        for seg in all_segment:
            tokens_a = topic_tokens
            b_length_cap = target_seq_length - len(tokens_a)
            tokens_b = seg[:b_length_cap]
            truncate_seq_pair(tokens_a, tokens_b, target_seq_length, self.rng)

            tokens, segment_ids = format_tokens_pair_n_segid(tokens_a, tokens_b)
            instance = SegmentInstance(
                tokens=tokens,
                segment_ids=segment_ids)
            instances.append(instance)
        return instances

    def write_instances(self, insts, output_file):
        return self.write_instance_to_example_files(insts, [output_file])
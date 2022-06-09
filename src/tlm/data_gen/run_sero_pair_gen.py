from functools import partial

from data_generator.job_runner import sydney_working_dir
from tf_util.record_writer_wrap import RecordWriterWrap
from tf_util.tf_logging import tf_logging
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.lm_datagen import UnmaskedPairGen
from tlm.data_gen.lm_worker import LMJobRunner
from tlm.data_gen.tf_logger_misc import log_print_inst


class SeroPairGen(UnmaskedPairGen):
    def __init__(self):
        super(SeroPairGen, self).__init__()

    def format_tokens_pair_n_segid(self, tokens_a, tokens_b):
        tokens = []
        segment_ids = []
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)
        return tokens, segment_ids

    def write_instances(self, instances, outfile):
        writer = RecordWriterWrap(outfile)
        example_numbers = []
        get_basic_input_features_fn = partial(get_basic_input_feature, self.tokenizer, self.max_seq_length)
        for (inst_index, instance) in enumerate(instances):
            features = get_basic_input_features_fn(instance.tokens, instance.segment_ids)
            features["use_context"] = create_int_feature([1])
            writer.write_feature(features)
            if inst_index < 20:
                log_print_inst(instance, features)
        writer.close()

        tf_logging.info("Wrote %d total instances", writer.total_written)
        return example_numbers

if __name__ == "__main__":
    working_dir = sydney_working_dir
    runner = LMJobRunner(1000, "sero_pair_gen", SeroPairGen)
    runner.start()

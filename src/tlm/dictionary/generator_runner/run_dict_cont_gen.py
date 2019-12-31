import os
import random

from cache import load_from_pickle
from data_generator import tokenizer_wo_tf as tokenization
from path import data_path
from tf_util.record_writer_wrap import RecordWriterWrap
from tf_util.tf_logging import tf_logging
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_int_feature


class GenerateDictContentWorker:
    def __init__(self, dictionary_pickle, max_word_tokens, max_seq_length, out_dir):
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.d = dictionary_pickle
        self.max_word_tokens = max_word_tokens
        self.max_seq_length = max_seq_length
        self.out_dir = out_dir

    def _create_instances(self, all_words):
        insts = []
        for word in all_words:
            def_list = self.d[word]
            word_tokens = self.tokenizer.tokenize(word)
            for def_tokens in def_list:
                def_tokens = def_tokens[:self.max_seq_length]
                segment_ids = [0] * len(def_tokens)
                r =  word_tokens, def_tokens, segment_ids
                insts.append(r)
        return insts

    def _write_instances(self, insts, output_file):
        writer = RecordWriterWrap(output_file)

        for instance in insts:
            word_tokens, def_tokens, segment_ids = instance
            word_tokens_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            features = get_basic_input_feature(self.tokenizer, self.max_seq_length, def_tokens, segment_ids)
            while len(word_tokens_ids) < self.max_word_tokens:
                word_tokens_ids.append(0)
            features["word"] = create_int_feature(word_tokens_ids)
            writer.write_feature(features)
        writer.close()
        tf_logging.info("Wrote %d total instances", writer.total_written)

    def work(self):
        all_words = list(self.d.keys())
        random.shuffle(all_words)
        cut = int(len(all_words) * 0.9)
        train_words = all_words[:cut]
        test_words = all_words[cut:]

        for words, job_id in [(train_words, "train"), (test_words, "test")]:
            insts = self._create_instances(words)
            output_file = os.path.join(self.out_dir, "{}".format(job_id))
            random.shuffle(insts)
            self._write_instances(insts, output_file)


if __name__ == "__main__":
    d = load_from_pickle("webster_parsed_w_cls")
    output_dir = os.path.join(data_path, "dict_def")
    g = GenerateDictContentWorker(d, 12, 96, output_dir)
    g.work()



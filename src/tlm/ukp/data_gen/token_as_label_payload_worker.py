import os

import data_generator
from data_generator import job_runner
from data_generator.argmining.ukp import DataLoader
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from tlm.data_gen.base import truncate_seq_pair
from tlm.data_gen.label_as_token_encoder import encode_label_and_token_pair
from tlm.data_gen.lm_datagen import UnmaskedPairedDataGen, SegmentInstance


class UkpTokenAsLabelGenerator(UnmaskedPairedDataGen):
    def __init__(self):
        super(UkpTokenAsLabelGenerator, self).__init__()
        self.ratio_labeled = 0.1  # Probability of selecting labeled sentence

    def create_instances(self, topic, labeled_data):
        topic_tokens = self.tokenizer.tokenize(topic.replace("_", " "))
        max_num_tokens = self.max_seq_length - 3 - len(topic_tokens)
        target_seq_length = max_num_tokens

        instances = []
        for label, tokens_b in labeled_data:
            tokens_a = []
            truncate_seq_pair(tokens_a, tokens_b, target_seq_length, self.rng)
            swap = False
            tokens, segment_ids = encode_label_and_token_pair(topic_tokens, label, tokens_b, tokens_a, swap)
            instance = SegmentInstance(
                tokens=tokens,
                segment_ids=segment_ids)
            instances.append(instance)

        return instances


class UkpTokenLabelPayloadWorker(job_runner.WorkerInterface):
    def __init__(self, out_path, generator):
        self.out_dir = out_path
        self.generator = generator

    def work(self, job_id):
        topic = data_generator.argmining.ukp_header.all_topics[job_id]
        ukp_data = self.get_ukp_dev_sents(topic)
        insts = self.generator.create_instances(topic, ukp_data)
        output_file = os.path.join(self.out_dir, topic.replace(" ", "_"))
        self.generator.write_instances(insts, output_file)

    def get_ukp_dev_sents(self, topic):
        loader = DataLoader(topic)
        data = loader.get_dev_data()
        tokenizer = get_tokenizer()

        def encode(e):
            sent, label = e
            tokens = tokenizer.tokenize(sent)
            return label, tokens

        label_sent_pairs = lmap(encode, data)
        return label_sent_pairs

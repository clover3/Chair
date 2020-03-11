import os

import data_generator.argmining
from data_generator import job_runner
from data_generator.argmining.ukp import DataLoader
from data_generator.common import get_tokenizer
from misc_lib import lmap, pick1, flatten
from tlm.data_gen.base import UnmaskedPairedDataGen, truncate_seq_pair, SegmentInstance
from tlm.data_gen.label_as_token_encoder import encode_label_and_token_pair
from tlm.ukp.sydney_data import sydney_get_ukp_ranked_list, load_tokens_for_topic


class UkpCluewebMixWorker(job_runner.WorkerInterface):
    def __init__(self, out_path, top_k, blind_topic, generator):
        self.out_dir = out_path
        self.generator = generator
        self.top_k = top_k
        self.blind_topic = blind_topic
        self.clueweb_token_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_tokens/"

    def work(self, job_id):
        topic = data_generator.argmining.ukp_header.all_topics[job_id]
        clueweb_docs = self.get_clueweb_docs(topic)
        ukp_data = self.get_ukp_sents(topic)
        insts = self.generator.create_instances(topic, clueweb_docs, ukp_data)
        output_file = os.path.join(self.out_dir, topic.replace(" ", "_"))
        self.generator.write_instances(insts, output_file)

    def get_clueweb_docs(self, topic):
        ranked_list = sydney_get_ukp_ranked_list()[topic]
        print("Ranked list contains {} docs, selecting top-{}".format(len(ranked_list), self.top_k))
        doc_ids = [doc_id for doc_id, _, _ in ranked_list[:self.top_k]]
        all_tokens = load_tokens_for_topic(self.clueweb_token_path, topic)
        docs = [all_tokens[doc_id] for doc_id in doc_ids if doc_id in all_tokens]
        return docs

    def get_ukp_sents(self, topic):
        if topic == self.blind_topic:
            return []

        loader = DataLoader(self.blind_topic)
        data = loader.get_topic_train_data(topic)
        tokenizer = get_tokenizer()

        def encode(e):
            sent, label = e
            tokens = tokenizer.tokenize(sent)
            return label, tokens

        label_sent_pairs = lmap(encode, data)
        return label_sent_pairs


class UkpCluewebMixGenerator(UnmaskedPairedDataGen):
    def __init__(self):
        super(UkpCluewebMixGenerator, self).__init__()
        self.ratio_labeled = 0.1  # Probability of selecting labeled sentence

    def create_instances(self, topic, raw_docs, labeled_data):
        # Format: [CLS] [Abortion] [LABEL_FAVOR] ...(ukp text)...[SEP] [ABORTION] [LABEL_UNK] ..(clue text).. [SEP]
        topic_tokens = self.tokenizer.tokenize(topic.replace("_", " "))
        # TODO iterate docs, pool chunk
        # randomly draw and sometimes insert labeled one
        # encode and add to instances
        max_num_tokens = self.max_seq_length - 3 - 2 - 2 * len(topic_tokens)
        target_seq_length = max_num_tokens
        docs_as_chunks, target_inst_num = self.pool_chunks_from_docs(raw_docs, target_seq_length)

        instances = []
        for _ in range(target_inst_num):
            chunk_1 = pick1(pick1(docs_as_chunks))

            m = self.rng.randint(1, len(chunk_1))
            tokens_a = flatten(chunk_1[:m])
            b_length = target_seq_length - len(tokens_a)
            if self.rng.random() < self.ratio_labeled and labeled_data:
                label, tokens_b = pick1(labeled_data)
            else:
                if self.rng.random() < 0.5 :
                    chunk_2 = pick1(pick1(docs_as_chunks))
                    tokens_b = flatten(chunk_2)[:b_length]
                else:
                    tokens_b = flatten(chunk_1[m:])[:b_length]
                label = -1
            truncate_seq_pair(tokens_a, tokens_b, target_seq_length, self.rng)

            swap = self.rng.random() < 0.5

            tokens, segment_ids = encode_label_and_token_pair(topic_tokens, label, tokens_b, tokens_a, swap)
            instance = SegmentInstance(
                tokens=tokens,
                segment_ids=segment_ids)
            instances.append(instance)

        return instances
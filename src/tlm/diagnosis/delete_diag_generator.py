from cache import *
from path import data_path
from collections import Counter
from data_generator import tokenizer_wo_tf as tokenization
import random
from tlm.wiki import bert_training_data as btd
from misc_lib import *


def truncate_seq(tokens_a, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a)
        if total_length <= max_num_tokens:
            break

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del tokens_a[0]
        else:
            tokens_a.pop()
    return tokens_a

class LMTrainGen:
    def __init__(self, out_path):
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

        self.masked_lm_prob = 0.15
        self.short_seq_prob = 0.1
        self.problem_per_job = 100 * 1000
        self.max_seq_length = 512
        self.max_predictions_per_seq = 20
        self.dupe_factor = 1
        self.out_path = out_path
        self.rng = random.Random(1)
        self.documents = None

    def load_document(self):
        self.documents = self._load_documents_from_pickle()

    def _load_documents_from_pickle(self):
        seg_id = self.rng.randint(0, 9)
        file_path = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/tokens/enwiki_train_tokens.{}"
        all_docs = []
        for j in range(100):
            full_id = seg_id * 100 + j
            f = open(file_path.format(full_id), "rb")
            all_docs.extend(pickle.load(f))
        return all_docs

    def pool_tokens(self, document, target_seq_length):
        results = []
        current_chunk = []
        current_length = 0
        max_num_tokens = self.max_seq_length - 2
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                tokens_a = flatten(current_chunk)
                tokens_a = truncate_seq(tokens_a, max_num_tokens, self.rng)
                results.append(tokens_a)
                current_chunk = []
                current_length = 0
            i += 1
        return results

    def create_instances_from_document(self, document):
        vocab_words = list(self.tokenizer.vocab.keys())
        max_num_tokens = self.max_seq_length - 2

        target_seq_length = max_num_tokens
        if self.rng.random() < self.short_seq_prob:
            target_seq_length = self.rng.randint(2, max_num_tokens)

        instances = []

        for raw_tokens in self.pool_tokens(document, target_seq_length):
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                segment_ids.append(0)

            tokens.append("[SEP]")
            segment_ids.append(0)

            (tokens, masked_lm_positions,
             masked_lm_labels) = btd.create_masked_lm_predictions(tokens,
                                                                  self.masked_lm_prob,
                                                                  self.max_predictions_per_seq, vocab_words, self.rng)
            instance = btd.TrainingInstance(
                tokens=tokens,
                segment_ids=segment_ids,
                is_random_next=False,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
            instances.append(instance)

        return instances



class DeleteTokenTestGen(LMTrainGen):
    def __init__(self, out_path, n_delete, target_words):
        super(DeleteTokenTestGen, self).__init__(out_path)
        self.target_words = target_words
        self.n_delete = n_delete
        NotImplemented


    def work(self):
        counter = Counter()
        # Generate test data for each tokens
        # Some tokens are frequent, some are rare

        for doc in self.documents:
            NotImplemented



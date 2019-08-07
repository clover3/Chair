import random
from tlm.tf_instance_writer import MaskedLmInstance, TrainingInstance

class TFInstanceMaker:
    def __init__(self, vocab_words):
        self.vocab_words = vocab_words
        self.rng = random.Random(0)

    def random_voca(self):
        return self.vocab_words[self.rng.randint(0, len(self.vocab_words) - 1)]

    def make_instance(self, problem):
        target, hint, mask = problem
        rng = self.rng
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in target:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in hint:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        output_tokens = list(tokens)
        masked_lms = []

        for index in mask:
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = self.random_voca()
            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        instance = TrainingInstance(
            tokens=output_tokens,
            segment_ids=segment_ids,
            is_random_next=False,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        return instance
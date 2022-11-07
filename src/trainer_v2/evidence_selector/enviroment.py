import random
from typing import List

from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.reinforce.monte_carlo_policy_function import SA


def dummy_environment(items: List[SA]) -> List[float]:
    return [random.random() for _ in items]


class PEPClient:
    def __init__(self):
        tokenizer = get_tokenizer()
        self.mask_id = tokenizer.wordpiece_tokenizer.vocab["[MASK]"]

    def request(self, items: List[SA]) -> List[float]:
        for state, action in items:
            input_ids, segment_ids = state
            select_mask: List[int] = action

            new_input_ids = []
            for t, m in zip(input_ids, select_mask):
                assert type(t) == int
                if m:
                    new_input_ids.append(t)
                else:
                    new_input_ids.append(self.mask_id)





    pass


def main():
    pass


if __name__ == "__main__":
    main()
from collections import Counter

from tlm.data_gen.lm_datagen import LMTrainGen


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



import random

PADDING = 0
OOV = 1


class SimpleLoader:
    def __init__(self, word2idx, X, y, seq_max):

        self.word2idx = word2idx
        self.train_X = X
        self.train_y = y

        self.seq_max = seq_max
        self.held_out = 64 * 4
        data = []
        cnt_in_V = 0
        cnt_oov = 0
        def convert(word):
            nonlocal cnt_in_V
            nonlocal cnt_oov
            if word in self.word2idx:
                cnt_in_V += 1
                return self.word2idx[word]
            else:
                cnt_oov += 1
                return OOV

        for i, x in enumerate(self.train_X):
            entry = []
            for token in x:
                entry.append(convert(token))

            entry = entry[:self.seq_max]
            while len(entry) < self.seq_max:
                entry.append(PADDING)
            y = self.train_y[i]
            data.append((entry, y))
        random.shuffle(data)
        self.all_data = data

        print("OOV rate : {}".format(cnt_oov / (cnt_oov + cnt_in_V)))


    def get_train_data(self):
        return self.all_data[self.held_out:]

    def get_dev_data(self):
        return self.all_data[:self.held_out]
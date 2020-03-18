import collections

import math

from list_lib import right


class LMClassifier:
    def __init__(self, P_w_C_dict, P_w_NC_dict):
        self.P_w_C_dict = P_w_C_dict
        self.P_w_NC_dict = P_w_NC_dict
        self.smoothing = 0.9

    def per_token_odd(self, token):
        smoothing = self.smoothing
        if token not in self.P_w_C_dict:
            return 0

        P_w_C = self.P_w_C_dict[token]
        P_w_NC = self.P_w_NC_dict[token]

        try:
            logC = math.log(P_w_C * smoothing + P_w_NC * (1 - smoothing))
            logNC = math.log(P_w_NC * smoothing + P_w_C * (1 - smoothing))
        except ValueError as e:
            print(P_w_C, P_w_NC)
            raise e
        return logC - logNC

    def counter_odd(self, counter):
        return sum([self.per_token_odd(key) * value for key, value in counter.items()])

    def tune_alpha(self, xy):
        vectors = []
        for x_i, y_i in xy:
            odd = self.counter_odd(x_i)
            vectors.append((odd, y_i))
        vectors.sort(key=lambda x: x[0], reverse=True)

        total = len(vectors)
        p = sum(right(xy))
        fp = 0
        max_acc = 0
        self.opt_alpha = 0
        for idx, (odd, label) in enumerate(vectors):
            alpha = odd - 1e-8
            if label == 0:
                fp += 1

            tp = (idx+1) - fp
            fn = p - tp
            tn = total - (idx+1) - fn
            acc = (tp + tn) / (total)
            if acc > max_acc:
                self.opt_alpha = alpha
                max_acc = acc

        print("Train acc : {}".format(max_acc))

    def predict(self, counter):
        return self.counter_odd(counter) > self.opt_alpha

    def counter_contribution(self, counter):
        return collections.Counter({key: self.per_token_odd(key) * cnt for key, cnt in counter.items()})

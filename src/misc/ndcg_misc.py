import numpy as np

r1 = np.array([[96.72353077, 96.71754058, 96.71380408, 96.70858749, 96.69818394, 96.69037012, 96.68298827, 96.65905576, 96.64763064, 96.62771171]])
r2 = np.array([[14041, 13414, 12543, 8936, 39832, 39832, 28731, 39604, 42599, 39832]])

r2_worst = np.sort(r2)
r2_best = np.flip(r2_worst)
print("This", r2)
print("Best", r2_best)
print("Worst", r2_worst)

from sklearn.metrics import ndcg_score, dcg_score


def show_ndcg():
    print("This {0:.4f}".format(ndcg_score(r1, r2, 4)))
    print("Best {0:.4f}".format(ndcg_score(r1, r2_best, 4)))
    print("Worst {0:.4f}".format(ndcg_score(r1, r2_worst, 4)))
    print(dcg_score(r1, r2))
    print(dcg_score(r1, r2_best))
    print(dcg_score(r1, r2_worst))

show_ndcg()


print("Assume the score without transfer learning was 96.6")
r1 = r1 - 96.7
print(r1)
show_ndcg()
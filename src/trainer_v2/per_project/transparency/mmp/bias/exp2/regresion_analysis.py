import sys
import statsmodels.api as sm
import numpy as np

from table_lib import tsv_iter

def main():
    record_d = {}
    unique_keys = set()
    for line in open(sys.argv[1], "r"):
        a, b, w_a, w_b = line.split()
        record_d[(a, b)] = int(w_a), int(w_b)
        unique_keys.add(a)
        unique_keys.add(b)

    score_table = {}
    for _car, item, score_s in tsv_iter(sys.argv[2]):
        score_table[item] = float(score_s)

    unique_keys = list(unique_keys)

    X_arr = []
    Y_arr = []
    for key1 in unique_keys:
        for key2 in unique_keys:
            if key1 == key2:
                continue

            k1 = score_table[key1]
            k2 = score_table[key2]
            x = k2 - k1
            w_1, w_2 = record_d[key1, key2]
            w2_rate = w_2 / (w_1 + w_2)
            y = w2_rate
            X_arr.append(x)
            Y_arr.append(y)

    # Sample data (replace these with your actual data)
    # X = np.array([K_A[i] - K_B[i] for i in range(100)])  # Difference between K_A and K_B
    # Y = np.array([1 if A_scores[i] > B_scores[i] else 0 for i in range(100)])  # 1 if A > B, 0 otherwise

    X = np.array(X_arr)
    Y = np.array(Y_arr)
    # Adding a constant to the model (for intercept)
    X = sm.add_constant(X)

    # Performing the linear regression
    model = sm.OLS(Y, X).fit()
    # Print the summary of the regression
    print(model.summary())





if __name__ == '__main__':
    main()
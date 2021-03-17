import scipy.stats as stats

x1 = [12, 2, 1, 12.2, 2.4]
x2 = [1, 4, 7, 1.5, 0]
tau, p_value = stats.kendalltau(x1, x2)
print(tau)

x1s = sorted(x1)
x2s = sorted(x2)
print(x1s)
print(x2s)
tau, p_value = stats.kendalltau(x1s, x2s)
print(tau)
x1 = list(range(6))
x2 = list(range(0, 12, 2))
tau, p_value = stats.kendalltau(x1, x2)
print(tau)

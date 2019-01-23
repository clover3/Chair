
import numpy as np

def label2logit(label, size):
    r = np.zeros([size,])
    r[label] = 1
    return r


def get_batches(data, batch_size):
    X, Y = data
    # data is fully numpy array here
    step_size = int((len(Y) + batch_size - 1) / batch_size)
    new_data = []
    for step in range(step_size):
        x = []
        y = []
        for i in range(batch_size):
            idx = step * batch_size + i
            if idx >= len(Y):
                break
            x.append(np.array(X[idx]))
            y.append(Y[idx])
        if len(y) > 0:
            new_data.append((np.array(x), np.array(y)))

    return new_data

def get_batches_ex(data, batch_size, n_inputs):
    # data is fully numpy array here
    step_size = int((len(data) + batch_size - 1) / batch_size)
    new_data = []
    for step in range(step_size):
        b_unit = [list() for i in range(n_inputs)]

        for i in range(batch_size):
            idx = step * batch_size + i
            if idx >= len(data):
                break
            for input_i in range(n_inputs):
                b_unit[input_i].append(data[idx][input_i])
        if len(b_unit[0]) > 0:
            batch = [np.array(b_unit[input_i]) for input_i in range(n_inputs)]
            new_data.append(batch)

    return new_data


def numpy_print(arr):
    return "".join(["{0:.3f} ".format(v) for v in arr])

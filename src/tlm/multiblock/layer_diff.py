from tensorflow.python import pywrap_tensorflow
import numpy as np


def load_values():
    save_path_string = "/mnt/nfs/work3/youngwookim/code/Chair/output/models/mr2/model.ckpt-295000"
    reader = pywrap_tensorflow.NewCheckpointReader(save_path_string)
    d_map = reader.get_variable_to_dtype_map()

    numpy_map = {}

    route_data = {}
    layer_data = {}

    for key in d_map:
        data = reader.get_tensor(key)
        numpy_map[key] = data
        tokens = key.split("/")
        if len(tokens) > 2 and tokens[2].startswith("layer_1_"):
            prefix ="layer_1_"
            route_no = tokens[2][len(prefix):]
            if route_no not in route_data:
                route_data[route_no] = {}

            post_fix = "/".join(tokens[3:])
            route_data[route_no][post_fix] = data
        elif len(tokens) > 2 and tokens[2].startswith("layer_"):
            if "mr_key" in key:
                continue
            layer_name = tokens[2]
            if layer_name not in layer_data:
                layer_data[layer_name] = {}

            post_fix = "/".join(tokens[3:])
            layer_data[layer_name][post_fix] = data

    def dist_sum(m1, m2):
        return np.sum(np.abs(m1 - m2))

    def dist_max(m1, m2):
        return np.max(np.abs(m1 - m2))

    def dist_avg(m1, m2):
        return np.average(np.abs(m1 - m2))

    route_keys = list(route_data.keys())
    n_route = len(route_keys)
    for i in range(n_route-1):
        route1 = route_keys[i]
        route2 = route_keys[i+1]
        print("  {}  vs {}".format(route1, route2))
        for key in route_data[route1]:
            if "adam_" in key:
                continue
            v1 = route_data[route1][key]
            v2 = route_data[route2][key]
            sum_d = dist_sum(v1, v2)
            avg_d = dist_avg(v1, v2)
            max_d = dist_max(v1, v2)

            print("{}\t{}\t{}\t{}\t{}".format(key, v1.shape, sum_d, avg_d, max_d))
        print("")
    print()

    layer_keys = list(layer_data.keys())
    n_layer = len(layer_keys)
    for i in range(n_layer-1):
        route1 = layer_keys[i]
        route2 = layer_keys[i+1]
        print(" {}  vs {}".format(route1, route2))
        for key in layer_data[route1]:
            if "adam_" in key:
                continue
            v1 = layer_data[route1][key]
            v2 = layer_data[route2][key]
            sum_d = dist_sum(v1, v2)
            avg_d = dist_avg(v1, v2)
            max_d = dist_max(v1, v2)

            print("{}\t{}\t{}\t{}\t{}".format(key, v1.shape, sum_d, avg_d, max_d))
        print("")
    print()


if __name__ == "__main__":
    load_values()


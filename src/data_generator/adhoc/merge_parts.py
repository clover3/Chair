import pickle


def merge(format_str, cont_range):
    result = []
    for elem in cont_range:
        pickle_path = format_str.format(elem)
        result += pickle.load(open(pickle_path, "rb"))

    return result


if __name__ == '__main__':
    format_str = "/mnt/nfs/work3/youngwookim/code/Chair/data/robust/payload_512_2k_encoded_parts/enc_payload512_part_{}"
    #format_str = "/mnt/nfs/work1/allan/youngwookim/payload_512_2k_encoded_parts/enc_payload512_part_{}"
    result = merge(format_str, range(0, 500000, 1000))
    pickle.dump(result, open("enc_payload_512.pickle", "wb"))
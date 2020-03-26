from list_lib import lfilter


def split_pos_neg(train):
    pos_insts = lfilter(lambda x: x['label'] == "1", train)
    neg_insts = lfilter(lambda x: x['label'] == "0", train)
    return neg_insts, pos_insts


def get_num_mention(data_point):
    num = data_point['num_mention']
    return num


def get_tf_from_datapoint(data_point):
    tf = data_point['feature']
    return tf


def get_label(data_point):
    return int(data_point['label'])


class DataPoint(dict):
    def __init__(self, d):
        super(DataPoint, self).__init__(d)


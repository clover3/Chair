from tlm.data_gen.robust_gen.robust_generators import RobustPosToNegRate
from tlm.robust.load import robust_query_intervals


def main():
    max_seq_length = 512
    st, ed = robust_query_intervals[0]
    query_list = [str(i) for i in range(st, ed + 1)]
    pos_neg_counter = RobustPosToNegRate(NotImplemented, max_seq_length, "desc", 1000)
    insts = pos_neg_counter.generate(query_list)


if __name__ == "__main__":
    main()


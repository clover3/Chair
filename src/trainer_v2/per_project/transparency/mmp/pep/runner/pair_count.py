from data_generator2.segmented_enc.es_mmp.pep_attn_common import iter_attention_mmp_pos_neg_paired


def main():
    partition_no = 0
    itr = iter_attention_mmp_pos_neg_paired(partition_no)
    print("{} items".format(sum(1 for _ in itr)))


if __name__ == "__main__":
    main()
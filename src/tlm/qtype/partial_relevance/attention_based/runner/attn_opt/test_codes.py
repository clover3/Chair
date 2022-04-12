from data_generator.tokenizer_wo_tf import JoinEncoder
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.segment_helper import get_always_active_mask
from alignment.data_structure.eval_data_structure import get_test_segment_instance


def test_get_always_active_mask():
    max_seq_length = 124
    inst = get_test_segment_instance()
    x3 = get_always_active_mask(inst, max_seq_length)
    join_encoder = JoinEncoder(max_seq_length)
    x0, x1, x2 = join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)
    for i1 in range(max_seq_length):
        print(x3[i1].tolist())

    print(len(inst.text1.tokens_ids))
    print(len(inst.text2.tokens_ids))
    CLS = join_encoder.CLS_ID
    SEP = join_encoder.SEP_ID
    for i1 in range(max_seq_length):
        for i2 in range(max_seq_length):

            # i1 or i2 is not valid location, than isi should not be always active
            if not x1[i1] or not x1[i2]:
                assert not x3[i1, i2]

            if x2[i1] == 0 and x2[i2] == 1:
                if x0[i1] not in [SEP, CLS] and x0[i2] not in [SEP, CLS]:
                    if x3[i1, i2]:
                        print(i1, i2)
                        print(x0[i1], x0[i2])
                    assert not x3[i1, i2]

            if x2[i1] == 1 and x2[i2] == 0:
                if x0[i1] not in [SEP, CLS] and x0[i2] not in [SEP, CLS]:
                    assert not x3[i1, i2]

    number_of_always_active = sum(x3)
    print(number_of_always_active)
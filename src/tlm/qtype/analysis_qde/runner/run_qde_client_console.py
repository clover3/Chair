from bert_api.client_lib import BERTClient
from cpath import pjoin, data_path, QDE_PORT
from data_generator.tokenizer_wo_tf import EncoderUnitPlain



def qde_console():
    max_seq_length = 512
    client = BERTClient("http://localhost", QDE_PORT, max_seq_length)

    voca_path = pjoin(data_path, "bert_voca.txt")
    q_encoder = EncoderUnitPlain(128, voca_path)
    d_encoder = EncoderUnitPlain(max_seq_length, voca_path)

    while True:
        query = input("Query: ")
        entity = input("Entity: ")
        doc = input("Document: ")

        qe_input_ids, qe_input_mask, qe_segment_ids = q_encoder.encode_pair(entity, query)
        de_input_ids, de_input_mask, de_segment_ids = d_encoder.encode_pair(entity, doc)
        one_inst = qe_input_ids, qe_input_mask, qe_segment_ids, de_input_ids, de_input_mask, de_segment_ids
        payload_list = [one_inst]
        ret = client.send_payload(payload_list)[0]
        for key in ['logits', 'd_bias', 'q_bias']:
            print(f"{key}: {ret[key]}")



if __name__ == "__main__":
    qde_console()

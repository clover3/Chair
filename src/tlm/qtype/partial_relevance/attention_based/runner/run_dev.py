from bert_api.client_lib import BERTClient
from data_generator.tokenizer_wo_tf import get_tokenizer, JoinEncoder
from port_info import MMD_Z_PORT
from tlm.qtype.partial_relevance.attention_based.bert_mask_predictor import get_bert_mask_predictor


def compare_with_mmd_z():
    query = 'Benefits of Coffee'
    doc = "Benefits of Coffee Coffee is good for concentration Coffee is delicious Coffee is good for health Coffee is not good"
    # doc = "Coffee is good for health "
    tokenizer = get_tokenizer()
    # client: BERTMaskClient = get_localhost_bert_mask_client()
    predictor = get_bert_mask_predictor()
    max_seq_length = 512
    mmd_client = BERTClient("http://localhost", MMD_Z_PORT, max_seq_length)
    score1 = mmd_client.request_single(query, doc)
    q_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query))
    print(q_tokens_ids)
    d_tokens = tokenizer.tokenize(doc)
    d_tokens_ids = tokenizer.convert_tokens_to_ids(d_tokens)

    join_encoder = JoinEncoder(max_seq_length)
    x0, x1, x2 = join_encoder.join(q_tokens_ids, d_tokens_ids)
    print(x0)
    print(x1)
    print(x2)
    new_payload = x0, x1, x2, {}
    score2 = predictor.predict([new_payload])
    print(score1, score2)


def main():

    pass



if __name__ == "__main__":
    main()


import numpy as np

from bert_api.client_lib import BERTClient
from cpath import pjoin, data_path
from data_generator.tokenizer_wo_tf import EncoderUnitPlain


def main():
    max_seq_length = 512
    client = BERTClient("http://localhost", 8128, max_seq_length)

    voca_path = pjoin(data_path, "bert_voca.txt")
    q_encoder = EncoderUnitPlain(128, voca_path)
    d_encoder = EncoderUnitPlain(max_seq_length, voca_path)

    def qde_eval(entity, query, doc):
        qe_input_ids, qe_input_mask, qe_segment_ids = q_encoder.encode_pair(entity, query)
        de_input_ids, de_input_mask, de_segment_ids = d_encoder.encode_pair(entity, doc)
        one_inst = qe_input_ids, qe_input_mask, qe_segment_ids, de_input_ids, de_input_mask, de_segment_ids
        payload_list = [one_inst]
        ret = client.send_payload(payload_list)[0]
        return ret

    doc = " buyer buyerbuy · er use buyer in a sentencebuyerone who buys ; consumera person whose work is to buy merchandise for a retail store webster ' s new world college dictionary , fifth edition copyright © 2014 by houghton mifflin harcourt publishing company . all rights reserved . link / citebuyernoun one that buys , especially a purchasing agent for a retail store . the american heritage® dictionary of the english language , fifth edition by the editors of the american heritage dictionaries . copyright © 2016 , 2011 by houghton mifflin harcourt publishing company . published by houghton mifflin harcourt publishing company . all rights reserved . link / citebuyer noun ( plural buyers ) a person who makes one or more purchases . every person who steps through the door is a potential buyer , so acknowledge their presence . ( retailing ) a person who purchases items for resale in a retail establishment . the supermarket ' s new buyer decided to stock a larger range of vegetarian foods . ( manufacturing ) a person who purchases items consumed or used as components in the manufacture of products . see also : buyer english wiktionary . available under cc - by - sa license . link / citebuyer - legal definitionn one who buys or agrees to make a purchase . see also purchaser . buyer in the ordinary course of business a person who buys goods in the usual manner from a person in the business of selling such goods and who does so in good faith and without knowledge that the sale violates another person ’ s ownership rights or security interest in the goods . such a buyer will have good title to the item purchased . see also holder in due course . ready , willing , and able buyer a person who is legally and financially able and has the disposition to make a particular purchase . straw buyer see straw person ( or man ) . webster ' s new world law dictionary copyright © 2010 by wiley publishing , inc . , hoboken , new jersey . used by arrangement with john wiley & sons , inc . link / cite"

    d0 = qde_eval("buyers",
                  "buyers meaning",
                  doc)
    doc_vector = d0['qtype_vector2']
    # Same Non-keyword Phrase "about how many different"

    adversary_r = qde_eval("buyers",
                  "about how many different buyers",
                  doc)

    adversary_q_vector = adversary_r['qtype_vector1']
    d2 = qde_eval("kinds of soil are there in the united states",
                  "about how many different kinds of soil are there in the united states",
                  doc)

    from_train_q_vector = d2['qtype_vector1']

    from_train_q_emb_dot_doc = np.dot(from_train_q_vector, doc_vector)
    adversary_q_emb_dot_doc = np.dot(adversary_q_vector, doc_vector)

    print("about how many different buyers", adversary_q_emb_dot_doc)
    print("about how many different kinds of soil are there in the united states", from_train_q_emb_dot_doc)


if __name__ == "__main__":
    main()
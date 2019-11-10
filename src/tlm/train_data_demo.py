from tlm.retrieve_lm.select_sentence import get_random_sent
from rpc.text_reader import TextReaderClient

def demo():
    n_repeat = 30
    robust_text = TextReaderClient()
    for j in range(n_repeat):
        row = get_random_sent()
        _, doc_id, loc, _, sent = row
        print('--------------')
        print(doc_id)
        print(sent)
        print("####")
        print(robust_text.retrieve(doc_id))






if __name__ == "__main__":
    demo()